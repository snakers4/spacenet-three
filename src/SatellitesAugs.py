import cv2
import time 
import types
import numpy as np
from numpy import random
import random
import imgaug as ia
from imgaug import augmenters as iaa
from torchvision import transforms
from PIL import Image
import torch
from skimage.transform import rotate

seed = 43
is_mask = False

# resnet and resnext means 
# 'mean': [0.485, 0.456, 0.406],
# 'std': [0.229, 0.224, 0.225]

# normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
#                                 std=[1, 1, 1, 1, 1, 1, 1, 1])

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

class SatellitesTrainAugmentation(object):
    def __init__(self,
                 shape=1280,
                 aug_scheme=None):
        
        if aug_scheme == True:
            print('Augmentations are enabled for train')
            self.augment_img = Compose([
                    # RandomCrop(800),
                    # NpyToPil(),
                    # transforms.Scale(shape),
                    # PilToNpy(),
                    ImgAugAugs(),
                    RandomCrop(shape),
                    ToTensor(),
                    normalize
                ])
            self.augment_mask = Compose([
                    # RandomCrop(800),
                    # NpyToPil(),
                    # transforms.Scale(shape),
                    # PilToNpy(),
                    ImgAugAugs(),
                    RandomCrop(shape),
                    ToTensor()
                ])            
        else:
            print('Augmentations are NOT enabled for train')
            self.augment_img = Compose([
                    RandomCrop(shape),                
                    # NpyToPil(),
                    # transforms.Scale(shape),
                    # PilToNpy(),
                    ToTensor(),
                    normalize
                ]) 
            self.augment_mask = Compose([
                    RandomCrop(shape),                
                    # NpyToPil(),
                    # transforms.Scale(shape),
                    # PilToNpy(),
                    ToTensor()
                ])               
        
    def __call__(self, img, mask):
        global seed
        global is_mask
        seed = random.randint(0,100)
        # process image
        is_mask = False
        # naive solution to working with 8-channel images 
        if img.shape[2]>3:
            img1 = self.augment_img(img[:,:,0:3]) 
            img2 = self.augment_img(img[:,:,3:6])
            img3 = self.augment_img(img[:,:,5:8])
            img = torch.cat((img1[0:3,:,:],img2[0:3,:,:],img3[1:3,:,:]))
        else:
            img = self.augment_img(img)
        # process mask
        is_mask = True
        # quick hack to evaluate paved only or non-paved only roads
        # mask = self.augment(mask[:,:,1:3])        
        mask = self.augment_mask(mask)
        return img,mask
class SatellitesTestAugmentation(object):
    def __init__(self,shape=1280,padding=6):
        self.augment_img = Compose([
                # NpyToPil(),
                # transforms.Pad(padding=padding, fill=0),
                RandomCrop(shape), # most likely causing low score on test test!          
                # NumpyPad(padding),
                # NpyToPil(),
                # transforms.Scale(shape),
                # PilToNpy(),            
                ToTensor(),
                normalize
            ])
        self.augment_mask = Compose([
                # NpyToPil(),
                # transforms.Pad(padding=padding, fill=0),
                RandomCrop(shape), # most likely causing low score on test test!          
                # NumpyPad(padding),
                # NpyToPil(),
                # transforms.Scale(shape),
                # PilToNpy(),            
                ToTensor()
            ])        
    def __call__(self, img, mask,seed_param=None):
        global is_mask
        global seed
        
        if seed_param is None:
            seed = random.randint(0,100)
        else:
            seed = seed_param

        # naive solution to working with 8-channel images 
        is_mask = False
        if img.shape[2]>3:
            img1 = self.augment_img(img[:,:,0:3]) 
            img2 = self.augment_img(img[:,:,3:6])
            img3 = self.augment_img(img[:,:,5:8])
            img = torch.cat((img1[0:3,:,:],img2[0:3,:,:],img3[1:3,:,:]))
        else:
            img = self.augment_img(img)
        if mask is not None:
            is_mask = True
            # quick hack to evaluate paved only or non-paved only roads
            # mask = self.augment(mask[:,:,1:3])
            mask = self.augment_mask(mask)            
        return img,mask
class SatellitesTestAugmentationPredict(object):
    def __init__(self,shape=1280,padding=6):
        self.augment_img = Compose([
                NumpyPad(padding),
                ToTensor(),
                normalize
            ])
        self.augment_mask = Compose([
                NumpyPad(padding),
                ToTensor()
            ])        
    def __call__(self, img, mask,seed_param=None):
        global is_mask
        global seed
        
        if seed_param is None:
            seed = random.randint(0,100)
        else:
            seed = seed_param

        # naive solution to working with 8-channel images 
        is_mask = False
        if img.shape[2]>3:
            img1 = self.augment_img(img[:,:,0:3]) 
            img2 = self.augment_img(img[:,:,3:6])
            img3 = self.augment_img(img[:,:,5:8])
            img = torch.cat((img1[0:3,:,:],img2[0:3,:,:],img3[1:3,:,:]))
        else:
            img = self.augment_img(img)
        if mask is not None:
            is_mask = True
            # quick hack to evaluate paved only or non-paved only roads
            # mask = self.augment(mask[:,:,1:3])
            mask = self.augment_mask(mask)            
        return img,mask
class SatellitesTestAugmentationTTA(object):
    def __init__(self,
                 padding=6,
                 hflip=False,
                 vflip=False):
        if (hflip == True and vflip == False):
            self.augment_img = Compose([
                    NpyToPil(),
                    transforms.Pad(padding=padding, fill=0),      
                    PilToNpy(),
                    HFlip(),            
                    ToTensor(),
                    normalize
                ])
            self.augment_mask = Compose([
                    NpyToPil(),
                    transforms.Pad(padding=padding, fill=0),
                    PilToNpy(),
                    HFlip(),
                    ToTensor()
                ])
        elif (hflip == False and vflip == True):
            self.augment_img = Compose([
                    NpyToPil(),
                    transforms.Pad(padding=padding, fill=0),      
                    PilToNpy(),
                    VFlip(),            
                    ToTensor(),
                    normalize
                ])
            self.augment_mask = Compose([
                    NpyToPil(),
                    transforms.Pad(padding=padding, fill=0),
                    PilToNpy(),
                    VFlip(),
                    ToTensor()
                ])            
        elif (hflip == True and vflip == True):
            self.augment_img = Compose([
                    NpyToPil(),
                    transforms.Pad(padding=padding, fill=0),      
                    PilToNpy(),
                    HFlip(),
                    VFlip(),           
                    ToTensor(),
                    normalize
                ])
            self.augment_mask = Compose([
                    NpyToPil(),
                    transforms.Pad(padding=padding, fill=0),
                    PilToNpy(),
                    HFlip(),
                    VFlip(),
                    ToTensor()
                ])
        else:
            self.augment_img = Compose([
                    NpyToPil(),
                    transforms.Pad(padding=padding, fill=0),      
                    PilToNpy(),
                    ToTensor(),
                    normalize
                ])
            self.augment_mask = Compose([
                    NpyToPil(),
                    transforms.Pad(padding=padding, fill=0),
                    PilToNpy(),
                    ToTensor()
                ])
            
    def __call__(self, img, mask,seed_param=None):
        global is_mask
        global seed
        
        if seed_param is None:
            seed = random.randint(0,100)
        else:
            seed = seed_param

        # naive solution to working with 8-channel images 
        is_mask = False
        if img.shape[2]>3:
            img1 = self.augment_img(img[:,:,0:3]) 
            img2 = self.augment_img(img[:,:,3:6])
            img3 = self.augment_img(img[:,:,5:8])
            img = torch.cat((img1[0:3,:,:],img2[0:3,:,:],img3[1:3,:,:]))
        else:
            img = self.augment_img(img)
        if mask is not None:
            is_mask = True
            # quick hack to evaluate paved only or non-paved only roads
            # mask = self.augment(mask[:,:,1:3])
            mask = self.augment_mask(mask)            
        return img,mask
class NumpyPad(object):
    def __init__(self,
                 padding = 6):
        self.padding = padding
    def __call__(self, img):
        return  np.pad(array=img,pad_width=((self.padding,self.padding), (self.padding,self.padding),(0,0)),mode='constant',constant_values=0) 
class HFlip(object):
    def __call__(self,
                 image):
        seq = iaa.Sequential([
            iaa.Fliplr(1.0), # horizontally flip 100% of all images
        ], random_order=False)       
        return seq.augment_image(image)
class VFlip(object):
    def __call__(self,
                 image):
        seq = iaa.Sequential([
            iaa.Flipud(1.0), # vertically flip 100% of all images
        ], random_order=False)       
        return seq.augment_image(image)    
# version compatible with 16-bit images
"""  
class ImgAugAugs(object):
    def __call__(self,
                 image):
        global seed        
        
        # poor man's flipping
        if seed%2==0:
            image = np.fliplr(image)
        elif seed%4==0:
            image = np.fliplr(image)
            image = np.flipud(image)
        
        # poor man's affine transformations
        image = rotate(image,
                     angle=seed,
                     resize=False,
                     clip=True,
                     preserve_range=True)        

        return image
"""
class ImgAugAugs(object):
    def __call__(self,
                 image):
        global seed        
        ia.seed(seed)
        seq = iaa.Sequential([
            # execute 0 to 1 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong            
            iaa.Fliplr(0.25), # horizontally flip 25% of all images
            iaa.Flipud(0.25), # vertically flip 25% of all images         
            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            iaa.Sometimes(0.25,            
                iaa.Affine(
                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    rotate=(-90, 90),
                    shear=(-5, 5)
                ),   
            ),
        ], random_order=True) # apply augmenters in random order        

        return seq.augment_image(image)
      
class RandomCrop(object):
    def __init__(self,
                 shape = 512):
        self.shape = shape
    def __call__(self,img):  
        global seed
        random.seed(seed)
        x_shift =  random.randint(0, img.shape[0] - self.shape - 1)
        random.seed(seed-1)
        y_shift =  random.randint(0, img.shape[1] - self.shape - 1)
        if len(img.shape)==3:
            return img[x_shift:x_shift+self.shape,x_shift:x_shift+self.shape,:]
        elif len(img.shape)==2:
            return img[x_shift:x_shift+self.shape,x_shift:x_shift+self.shape]
        else:
            raise NotImplementedError
class Normalize(object):
    def __init__(self,mean,std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
    def __call__(self, image):
        image = image.astype(np.float32)
        image[:,:,0:3] -= self.mean
        image[:,:,0:3] *= (1/self.std)
        return image.astype(np.float32)
class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img    
class ToCV2Image(object):
    def __call__(self, tensor):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0))
class PilToNpy(object):
    def __call__(self, pil_image):
        return np.array(pil_image)    
class NpyToPil(object):
    def __call__(self, cv2_image):
        return Image.fromarray(cv2_image)  
class ToTensor(object):
    def __call__(self, cvimage):
        global is_mask
        # process masks
        if (is_mask==True) and (len(cvimage.shape)==2):
            cvimage = np.expand_dims(cvimage, 2)
            cvimage = (cvimage > 255 * 0.5).astype(np.uint8)
            return torch.from_numpy(cvimage).permute(2, 0, 1).float()
        elif (is_mask==True) and (len(cvimage.shape)==3):
            cvimage = (cvimage > 255 * 0.5).astype(np.uint8)
            return torch.from_numpy(cvimage).permute(2, 0, 1).float()            
        else:
            # process images
            try:
                return torch.from_numpy(cvimage).permute(2, 0, 1).float().div(255)
            except:
                return torch.from_numpy((cvimage.transpose((2, 0, 1))).copy()).float().div(255)
class CannyEdges(object):
    def __init__(self,threshold1=100,threshold2=200):
        self.threshold1 = threshold1
        self.threshold2 = threshold2
    def __call__(self, image):
        canny_region = np.uint8(image[:,:,0:3])
        edges = cv2.Canny(canny_region,self.threshold1,self.threshold2)
        return np.dstack( ( image,edges) ) 
class SaliencyMap(object):
    def __call__(self, image):
        sm = pySaliencyMap(image.shape[0], image.shape[1])
        return np.dstack( ( image,sm.SMGetSM(image)) ) 
class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image):
        im = image.copy()
        im = self.rand_brightness(im)
        if random.randint(1,2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im = distort(im)
        return self.rand_light_noise(im)
class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image):
        if random.randint(1,2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image
class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image):
        if random.randint(1,2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image
class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image):
        if random.randint(1,2):
            swap = self.perms[random.randint(0,len(self.perms)-1)]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image
class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image
class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image):
        if random.randint(1,2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image
class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image):
        if random.randint(1,2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image    
class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32)
class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image
