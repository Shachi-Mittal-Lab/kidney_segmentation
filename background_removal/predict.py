from torch.autograd import Variable
import torch
from skimage import io

def predict(vgg, pred_loader):
    print("Predicting")
    print("-" * 10)
    batches = len(pred_loader)
    img_files = []
    predictions = []

    for i, data in enumerate(pred_loader):
        global batchnum
        batchnum = i
        if i % 100 == 0:
            print("\rPredict batch {}/{}".format(i, batches), end="", flush=True)

        # Do not train, set to evaluation mode
        vgg.train(False)
        vgg.eval()
        inputs, labels, filenames = data

        # Formatting for GPU vs CPU
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # Do not calc gradients
        with torch.no_grad():
            outputs = vgg(inputs)

        _, preds = torch.max(outputs.data, 1)

        img_files.extend(filenames)
        predictions.extend(preds.tolist())

        # Clear memory
        del inputs, labels, outputs, preds
        torch.cuda.empty_cache()

    return img_files, predictions