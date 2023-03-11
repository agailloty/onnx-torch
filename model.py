import torch

best_model = torch.load("bestmodel.pt")

# NCHW (batch size, number of channels, height, width).
x = torch.randn(15, 3, 224, 224, requires_grad=True).cuda()

torch.onnx.export(best_model,
                 x,
                 "bestmodel.onnx",
                 verbose=False,
                 input_names=["actual_input"],
                 output_names=["output"],
                 export_params=True,
                 )
