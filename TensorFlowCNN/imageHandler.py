from PIL import Image,ImageOps
import glob
size = ((28,28))
class ImageHandler():
    #取训练图片数据集 (这里是用自己的切图方式切出来然后手动分类的)
    def get_trainning_images():
        dataset = []
        for i in range(10):
            all_pngs = glob.glob(r'.\trainningImages\{i}\*.png'.format(i=i))#打开所有i的训练图片
            for imgdir in all_pngs: 
                result = [0 for i in range(10)]
                result[i] = 1 #构造结果数组 方便训练输出结果 因为网络需要输出所有可能的概率 在这里就是0到9的概率所以需要构建一个数组。
                img = Image.open(imgdir).resize(size) #重构一下图片大小
                imgdata = img.getdata() #数值化
                newdata = list(map(lambda x :0.5 if x == 255 else -0.5,imgdata)) #归一化处理 把像素为255的数据处理为0.5 其他处理为-0.5方便计算
                dataset.append((newdata,result))
        return dataset #数据集结构为(图片二值化数组,图片对应的正确数字)
    #Y轴像素值求和 数组
    def Image_to_Y_sum_array(image):
        array = []
        for h in range(image.height):
            tmpsum = 0
            for w in range(image.width):
                tmpsum +=image.getpixel((w,h))
            array.append(tmpsum)
        return array
    #X轴像素值求和 数组
    def Image_to_X_sum_array(image):
        array = []
        for w in range(image.width):
            tmpsum = 0
            for h in range(image.height):
                tmpsum +=image.getpixel((w,h))
            array.append(tmpsum)
        return array
    #取图像断点数组
    def get_break_point(imageXarray):
        points = []
        for i in range(1,len(imageXarray)):
            first = imageXarray[i - 1]
            next = imageXarray[i]
            if ((first > next and first == 25500) or (first < next and next == 25500)):
                points.append(i - 1)
        return points
    #图片字符切分 初次切分判断像素和的波动 切分出来波谷部分
    def crop_Image(self,image):
        imageXarray = self.Image_to_X_sum_array(image)
        breakPoints = self.get_break_point(imageXarray)
        cropedImages = []
        for i in range(0,int(len(breakPoints) / 2)):
            cropedImages.append(image.crop((breakPoints[i * 2],0,breakPoints[i * 2 + 1],image.height)))
        if (len(cropedImages) == 0):
            return None
        while (len(cropedImages) < 4):
                cropedImages = self.reCut(cropedImages)
        cropedImages = self.del_white_bar(self,cropedImages)
       # images = self.resize_images(images)
        return cropedImages
    #去除白条
    def del_white_bar(self,images):
        newImages = []
        for image in images:
            yArray = self.Image_to_Y_sum_array(image)
            for y in range(len(yArray) - 1):
                if (yArray[y] == 255 * image.width and yArray[y + 1] < 255 * image.width):
                    topY = y
                    break
            for y in reversed(range(len(yArray) - 1)):  
                  if (yArray[y + 1] == 255 * image.width and yArray[y] < 255 * image.width):
                    bottomY = y
                    break
            newImages.append(image.crop((0,topY,image.width,bottomY)))
        return newImages
    #针对初次切分无法分割为4份的图像 强行切分最宽的一个字符 五五开
    def reCut(cropedImages):
        newImages = sorted(cropedImages,key=lambda x:x.width)
        maxImage = newImages.pop()
        index = cropedImages.index(maxImage)
        cropedImages.remove(maxImage)
        cropedImages.insert(index,maxImage.crop((maxImage.width / 2,0,maxImage.width,maxImage.height)))
        cropedImages.insert(index,maxImage.crop((0,0,maxImage.width / 2,maxImage.height)))
        return cropedImages
    #二值化图片数组
    def Image_binaryzation(images):
        imglist = []
        for i in images:
            i = i.convert('L')
            pixdata = i.load()
            w, h = i.size
            for y in range(h):
                for x in range(w):
                    if pixdata[x, y] < 140:
                        pixdata[x, y] = 0
                    else:
                        pixdata[x, y] = 255
            #i.convert('L')
            #for w in range(i.width):
            #    for h in range(i.height):
            #        i.putpixel((w,h),255 if i.getpixel((w,h))>150 else 0)
            #       # i.putpixel((w,h),0)
            imglist.append(i)
        return imglist
    #def binarizing(img,threshold): #input: gray image
    #    pixdata = img.load()
    #    w, h = img.size
    #    for y in range(h):
    #        for x in range(w):
    #            if pixdata[x, y] < threshold:
    #                pixdata[x, y] = 0
    #            else:
    #                pixdata[x, y] = 255
    #    return img
    
    #读入图片数组
    def read_images(start_index,end_index):
        imglist = []
        for i in range(start_index,end_index):
            img = Image.open(r'.\testCodeImages\{i}.png'.format(i=i))
            imglist.append(img)
        return imglist
    
    def resize_images(images):
        ret = []
        for img in images:
              ret.append(img.resize(size))
        return ret
    #数值化图片
    def getdata(images):
        ret = []
        for img in images:
            ret.append(img.getdata())
        return ret
    def normalalizeImages(Images):
        ret = []
        for img in Images:
            ret.append(list(map(lambda x: .5 if x == 255 else -.5,img)))
        return ret