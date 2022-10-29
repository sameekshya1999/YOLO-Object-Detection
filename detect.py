from __future__ import division
import torch 
from torch.autograd import Variable
import cv2 
from util import *
from DNModel import net as Darknet
from img_process import inp_to_image, custom_resize




confidence = 0.5
nms_thesh = 0.4
start = 0
CUDA = torch.cuda.is_available()
num_classes = 80

bbox_attrs = 5 + num_classes

print("Loading network")
model = Darknet("cfg/yolov3.cfg")
model.load_weights("yolov3.weights")
print("Network loaded")
classes = load_classes('cfg/coco.names')
model.DNInfo["height"] = 128
inp_dim = int(model.DNInfo["height"])
model.eval()
lbls = 0



def prepare_input(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    Perform tranpose and return Tensor
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (custom_resize(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def write(x, img):
	global lbls
	c1 = tuple(x[1:3].int())
	c2 = tuple(x[3:5].int())
	cls = int(x[-1])
	label = "{0}".format(classes[cls])
	#(width, height) = (640, 480)
	if (c1[0], c1[1]) >= (10, 10):
		
		print("label=>{}".format(label))
		cv2.rectangle(img, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), (0, 255, 0), 1)
			
		t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
		c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
		if c2[0] == 0:
			return img
		cv2.rectangle(img, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), (0, 0, 255), -1)
		cv2.putText(img, label, (int(c1[0]), int(c1[1]) + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
	else:
		pass
	return img


def getvidanddetect():
	img = None
	cap = cv2.VideoCapture(0)
	while True:
		_, image = cap.read()
		im = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		#Detection phase
		img, orig_im, dim = prepare_input(im, inp_dim)
		im_dim = torch.FloatTensor(dim).repeat(1,2) 
		with torch.no_grad():   
			output = model(Variable(img), CUDA)
		output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)
		if type(output) == int:
			cv2.imshow("Image", orig_im)
			key = cv2.waitKey(1)
			if key & 0xFF == ord('x'):
				break
			continue
		im_dim = im_dim.repeat(output.size(0), 1)
		scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)

		output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
		output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2

		output[:,1:5] /= scaling_factor

		for i in range(output.shape[0]):
			output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
			output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])

		list(map(lambda x: write(x, orig_im), output))
		cv2.imshow('Video',orig_im)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	getvidanddetect()

    
    

