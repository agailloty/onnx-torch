import torch

best_model = torch.load("bestmodel.pt")

# NCHW (batch size, number of channels, height, width).
x = torch.randn(1, 3, 224, 224, requires_grad=True).cuda()

torch.onnx.export(best_model,
                 x,
                 "bestmodel.onnx",
                 verbose=False,
                 input_names=["ModelInput"],
                 output_names=["Output"],
                 export_params=True,
                 )
