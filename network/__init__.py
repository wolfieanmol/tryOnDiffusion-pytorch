from network.parallel_unet import ParallelUnetModel128


def get_model():
    model = ParallelUnetModel128()
    return model


if __name__ == "__main__":
    model = get_model()
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    # print(model)
