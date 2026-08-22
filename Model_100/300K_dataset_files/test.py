import os
import torch

from model import (
    ModelCNN_fvs,
    ModelResNet34_fvs,
    ModelResNet50_fvs,
    ModelDenseNet121_fvs,
    ModelSwinT_fvs,
    ModelEfficientNetB0_fvs,
    ModelCustomCNN_fvs
)
from Call_dataset import MyDataset
from save_validation import save_prediction

# --- Device configuration ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# Mapping between a keyword found in the .pth filename and its architecture class.
ARCH_KEYWORDS = [
    ("modelcnn", ModelCNN_fvs),
    ("resnet34", ModelResNet34_fvs),
    ("resnet50", ModelResNet50_fvs),
    ("densenet121", ModelDenseNet121_fvs),
    ("efficientnetb0", ModelEfficientNetB0_fvs),
    ("swint", ModelSwinT_fvs),
    ("CustomCNN", ModelCustomCNN_fvs),
]

def build_model_from_filename(pth_filename, in_instances, in_channels):
    """
    Infer the architecture from the .pth filename and instantiate the model.

    Returns (model, architecture_name) or (None, None) if no keyword matches.
    """
    name = pth_filename.lower()
    for keyword, model_class in ARCH_KEYWORDS:
        if keyword in name:
            return model_class(in_instances, in_channels), model_class.__name__
    return None, None


def find_all_pth_files(root):
    """
    Recursively collect every .pth file under `root` (all subfolders).
    """
    pth_files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith(".pth"):
                pth_files.append(os.path.join(dirpath, f))
    return sorted(pth_files)


def build_test_dataset(data_folder, in_instances, in_channels, case):
    """
    Build the test dataset once (it is identical for every model).

    case = "synthetic": target available -> returned with the inputs.
    case = "measured" : real field data, no ground truth.
    """
    input_folder = os.path.join(data_folder, "input")
    if case == "synthetic":
        dataset = MyDataset(in_instances, in_channels,
                            input_folder,
                            os.path.join(data_folder, "output"))
        has_target = True
    else:
        dataset = MyDataset(in_instances, in_channels, input_folder)
        has_target = False
    return dataset, has_target


def run_inference_on_dataset(model, dataset, in_instances, folder_save_result, has_target):
    """
    Run the model on every sample and save the prediction files.
    """
    model.eval()
    os.makedirs(folder_save_result, exist_ok=True)

    with torch.no_grad():
        for idx, data in enumerate(dataset):
            if has_target:
                *all_inputs, target = data
                target = target.unsqueeze(0).to(DEVICE)
            else:
                all_inputs = list(data)
                target = None

            # Add the batch dimension and move to device
            all_inputs = [inp.unsqueeze(0).to(DEVICE) for inp in all_inputs]
            predict = model(*all_inputs)

            # Map tensors back to their instance names (robust to any in_instances order)
            named = dict(zip(in_instances, all_inputs))
            input_fvs = named['fvs'].squeeze(0).cpu() if 'fvs' in named else None
            input_x0 = named['x0'].flatten().cpu() if 'x0' in named else None
            input_dx = named['dx'].flatten().cpu() if 'dx' in named else None
            input_Ch = named['Ch'].flatten().cpu() if 'Ch' in named else None

            save_prediction(
                input_fvs=input_fvs,
                predict=predict.squeeze(0).cpu(),
                file_id=idx,
                folder_save_result=folder_save_result,
                input_x0=[input_x0[0]] if input_x0 is not None else None,
                input_dx=[input_dx[0]] if input_dx is not None else None,
                input_Ch=[input_Ch[0]] if input_Ch is not None else None,
                target=target.squeeze(0).cpu() if target is not None else None,
            )


def run_test(pth_root, data_folder, in_instances, in_channels, case="measured"):
    """
    Test every .pth model found under `pth_root` on the test data.

    Results are saved under:
        <data_folder>/predict/<pth subfolder>/<pth name>/
    so each model gets its own result folder and nothing is overwritten.
    """
    dataset, has_target = build_test_dataset(data_folder, in_instances, in_channels, case)
    pth_files = find_all_pth_files(pth_root)
    print(f"Found {len(pth_files)} .pth files under '{pth_root}'")
    print(f"Test data: '{data_folder}' ({len(dataset)} samples, case = {case})")
    print(f"Device: {DEVICE}\n")

    summary = []
    for pth_path in pth_files:
        pth_name = os.path.basename(pth_path)
        model, arch_name = build_model_from_filename(pth_name, in_instances, in_channels)

        if model is None:
            print(f"[SKIP] Unknown architecture for '{pth_name}'")
            summary.append((pth_path, "SKIPPED (unknown architecture)"))
            continue

        # Keep the PTH subfolder structure in the output path
        rel_dir = os.path.relpath(os.path.dirname(pth_path), pth_root)
        model_tag = os.path.splitext(pth_name)[0]
        folder_save_result = os.path.join(data_folder, "predict", rel_dir, model_tag)

        print(f"[TEST] {arch_name:25s} <- {pth_path}")
        try:
            state_dict = torch.load(pth_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state_dict)
            model.to(DEVICE)

            run_inference_on_dataset(model, dataset, in_instances, folder_save_result, has_target)

            print(f"       -> results saved in: {folder_save_result}")
            summary.append((pth_path, "OK"))
        except Exception as e:
            # One failing checkpoint must not stop the whole batch
            print(f"[FAIL] {pth_name}: {e}")
            summary.append((pth_path, f"FAILED: {e}"))
        finally:
            # Free GPU memory before loading the next model
            del model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    # --- Final summary ---
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    n_ok = sum(1 for _, s in summary if s == "OK")
    for path, status in summary:
        print(f"  [{status}] {path}")
    print(f"\n{n_ok}/{len(summary)} models tested successfully.")

if __name__ == "__main__":
    run_test(
        pth_root="PTH",
        data_folder="testing_measured",
        in_instances=['fvs', 'x0', 'dx', 'Ch'],
        in_channels=3,
        case="measured",
    )