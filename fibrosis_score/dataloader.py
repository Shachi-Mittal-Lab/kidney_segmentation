import torchvision
# Custom dataloader that includes image file paths, image label, and image. 
# Extends torchvision.datasets.ImageFolder

class ImageFolderWithPaths(torchvision.datasets.ImageFolder):

    # Override the __getitem__ method. This is the method that dataloader calls.
    def __getitem__(self, index):
        # This is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # The image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = original_tuple + (path,)
        return tuple_with_path