import torch
import numpy as np
import gradio as gr

from PIL import Image
from models import dehazeformer


def infer(raw_image):
	network = dehazeformer()
	network.load_state_dict(torch.load('./saved_models/dehazeformer.pth', map_location=torch.device('cpu'))['state_dict'])
	# torch.save({'state_dict': network.state_dict()}, './saved_models/dehazeformer.pth')

	network.eval()
	
	image = np.array(raw_image, np.float32) / 255. * 2 - 1
	image = torch.from_numpy(image)
	image = image.permute((2, 0, 1)).unsqueeze(0)

	with torch.no_grad():
		output = network(image).clamp_(-1, 1)[0] * 0.5 + 0.5	
		output = output.permute((1, 2, 0))
		output = np.array(output, np.float32)
		output = np.round(output * 255.0)

	output = Image.fromarray(output.astype(np.uint8))

	return output


title = "Dehazer"
description = f""
examples = [
		["examples/1.jpg"],
		["examples/2.jpg"],
		["examples/3.jpg"],
		["examples/4.jpg"],
		["examples/5.jpg"],
		["examples/6.jpg"]
]

iface = gr.Interface(
	infer,
	inputs="image", outputs="image",
	title=title,
	description=description,
	allow_flagging='never',
	# examples=examples,
)
iface.launch(share=True)