import os
import sys
import torch
import torchvision
import torchaudio

def smoke_test_cuda() -> None:
    gpu_arch_ver = os.environ['GPU_ARCH_VER']
    gpu_arch_type = os.environ['GPU_ARCH_TYPE']
    is_cuda_system = gpu_arch_type == "cuda"

    if(not torch.cuda.is_available() and is_cuda_system):
        print(f"Expected CUDA {gpu_arch_ver}. However CUDA is not loaded.")
        sys.exit(1)
    if(torch.cuda.is_available()):
        if(torch.version.cuda != gpu_arch_ver):
            print(f"Wrong CUDA version. Loaded: {torch.version.cuda} Expected: {gpu_arch_ver}")
            sys.exit(1)
        y=torch.randn([3,5]).cuda()
        print(f"torch cuda: {torch.version.cuda}")
        #todo add cudnn version validation
        print(f"torch cudnn: {torch.backends.cudnn.version()}")

def smoke_test_torchvision() -> None:
    import torchvision.datasets as dset
    import torchvision.transforms
    from torchvision.io import read_file, decode_jpeg, decode_png
    print('Is torchvision useable?', all(x is not None for x in [torch.ops.image.decode_png, torch.ops.torchvision.roi_align]))
    img_jpg = read_file('./assets/rgb_pytorch.jpg')
    img_jpg_nv = decode_jpeg(img_jpg)
    img_png = read_file('./assets/rgb_pytorch.png')
    img__png_nv = decode_png(img_png)

def smoke_test_vision_1() -> None:
    from torchvision.io import read_image
    from torchvision.models import resnet50, ResNet50_Weights

    img = read_image("test/assets/encode_jpeg/grace_hopper_517x606.jpg")

    # Step 1: Initialize model with the best available weights
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0)

    # Step 4: Use the model and print the predicted category
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"{category_name}: {100 * score:.1f}%")


def smoke_test_torchaudio() -> None:
    import torchaudio.compliance.kaldi  # noqa: F401
    import torchaudio.datasets  # noqa: F401
    import torchaudio.functional  # noqa: F401
    import torchaudio.models  # noqa: F401
    import torchaudio.pipelines  # noqa: F401
    import torchaudio.sox_effects  # noqa: F401
    import torchaudio.transforms  # noqa: F401
    import torchaudio.utils  # noqa: F401


def main() -> None:
    #todo add torch, torchvision and torchaudio tests
    print(f"torch: {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    print(f"torchaudio: {torchaudio.__version__}")
    smoke_test_cuda()
    smoke_test_torchvision()
    smoke_test_torchaudio()

if __name__ == "__main__":
    main()
