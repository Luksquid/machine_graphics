import cv2
from typing import Optional
from math import *
from matplotlib.image import *
import matplotlib.pyplot as plt
from enum import Enum

class ColorModel(Enum):
    rgb = 0
    hsv = 1
    hsi = 2
    hsl = 3
    gray = 4  # 2d picture


class ImageDiffMethod(Enum):
    mse = 0
    rmse = 1

class BaseImage:
    data: np.ndarray  
    color_model: ColorModel 
    path: str

    def __init__(self, path: str) -> None:
        self.path = path
        self.color_model: int = 0
        if (path != None):
            self.data = imread(path)
        pass

    def save_img(self, path: str) -> None:
        imsave(path, self.data)
        pass

    def show_img(self) -> None:
        cv2.imshow('image', self.data)
        cv2.waitKey(0)
        pass

    def get_layer(self, layer_id: int):
        match layer_id:
            case 0:
                return self.data[:, :, 0]
            case 1:
                return self.data[:, :, 1]
            case 2:
                return self.data[:, :, 2]
        pass

class GrayScaleTransform(BaseImage):
    img: BaseImage

    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.img = BaseImage(path)

    def to_gray(self) -> BaseImage:

        redLayer = self.img.get_layer(0)
        greenLayer = self.img.get_layer(1)
        blueLayer = self.img.get_layer(2)
        res = np.arange(self.img.data.shape[0] * self.img.data.shape[1]).reshape(self.img.data.shape[0], self.img.data.shape[1])
        for i in range(redLayer.shape[0]):
            for j in range(redLayer.shape[1]):
                res[i, j] = (float(redLayer[i, j]) + float(greenLayer[i, j]) + float(blueLayer[i ,j])) /3
        result = BaseImage(None)
        result.color_model = 4
        result.data = res.astype(np.uint8)
        return result

    def to_sepia(self, alpha: float=None, beta: float = None, w: int = None) -> BaseImage:

        greyV = self.to_gray().data
        res = np.arange(3 * self.img.data.shape[0] * self.img.data.shape[1]).reshape(self.img.data.shape[0],self.img.data.shape[1], 3)
        for i in range(greyV.data.shape[0]):
            for j in range(1, greyV.data.shape[1]):
                L0 = (greyV.data[i, j])
                L1 = (greyV.data[i, j])
                L2 = (greyV.data[i, j])
                if (alpha!=None and beta!=None):
                    L0 = L0 * alpha
                    L1 = L1
                    L2 = L2 * beta
                else:
                    L0 = L0 + 2 * w
                    L1 = L1 + w
                    L2 = L2
                res[i, j] = np.array([L0, L1, L2])
                for k in range(3):
                     if(res[i,j,k]>255):
                         res[i, j, k] = 255
        Result = BaseImage(None)
        Result.data = res.astype(np.uint8)
        Result.color_model = 0
        return Result

class Histogram:
    values: np.ndarray

    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.valuesOfHistogram = []
        pass

    def calculateValues(self):

        match self.values.ndim:

            case 2:
                    res = np.arange(256)
                    for i in range(256):
                        count_arr = np.bincount(self.values.reshape(self.values.shape[0] * self.values.shape[1]))
                        try:
                            res[i] = count_arr[i]
                        except:
                            res[i] = 0
                    return res
            case 3:
                    res1 = np.arange(256)
                    res2 = np.arange(256)
                    res3 = np.arange(256)
                    reds = self.values[:, :, 0]
                    greens = self.values[:, :, 1]
                    blues = self.values[:, :, 2]
                    for i in range(256):
                        count_arr1 = np.bincount(reds.reshape(reds.shape[0] * reds.shape[1]))
                        count_arr2 = np.bincount(greens.reshape(greens.shape[0] * greens.shape[1]))
                        count_arr3 = np.bincount(blues.reshape(blues.shape[0] * blues.shape[1]))
                        try:
                            res1[i] = count_arr1[i]
                        except:
                            res1[i] = 0
                        try:
                            res2[i] = count_arr2[i]
                        except:
                            res2[i] = 0
                        try:
                            res3[i] = count_arr3[i]
                        except:
                            res3[i] = 0

                    return np.array([res1, res2, res3])



    def plot(self) -> None:

        if(np.any(self.valuesOfHistogram)==False):
            hist = self.calculateValues()
        else:
            hist = self.valuesOfHistogram
        match self.values.ndim:

            case 2:
                plt.plot(hist, '#5c5c5c')
                plt.show()
            case 3:
                plt.subplots_adjust(wspace=1)
                plt.subplot(1, 3, 1)
                plt.plot(hist[0], '#cc0000')
                plt.subplot(1, 3, 2)
                plt.plot(hist[1], '#2ccc00')
                plt.subplot(1, 3, 3)
                plt.plot(hist[2], '#1f00cc')
                plt.show()

    def to_cumulated(self) -> 'Histogram':
        hV = self.calculateValues()
        match hV.ndim:
            case 1:
                sum = 0
                h = np.zeros(256)
                for i in range(256):
                    sum = sum + hV[i]
                    hV[i] = sum
                res = Histogram(self.values)
                res.valuesOfHistogram = hV
                return res
            case 2:
                hist = [[], [], []]
                sum = [0, 0, 0]
                for i in range(255):
                    sum[0] = sum[0] + hV[0][i]
                    sum[1] = sum[1] + hV[1][i]
                    sum[2] = sum[2] + hV[2][i]
                    hist[0].append(sum[0])
                    hist[1].append(sum[1])
                    hist[2].append(sum[2])
                res = Histogram(self.values)
                res.valuesOfHistogram = [hist[0], hist[1], hist[2]]
                return res
        pass

class ImageComparison(BaseImage):

    def __init__(self, path: str):
        super().__init__(path)

    def histogram(self) -> Histogram:
        g = GrayScaleTransform(self.path)
        hist = Histogram(g.to_gray().data)
        return hist
        pass

    def compare_to(self, other: BaseImage, method: ImageDiffMethod) -> float:
        # mse 0
        # rmse 1
        h1 = self.histogram().calculateValues()
        g = GrayScaleTransform(other.path)
        h2 = Histogram(g.to_gray().data)
        h2 = h2.calculateValues()
        match method:
            case 0:
                return np.sum((np.power((h1 - h2), 2)) / 255)
            case 1:
                return np.sqrt(np.sum((np.power((h1 - h2), 2)) / 255))
        pass

class ImageAligning(BaseImage):

    def __init__(self, path: str):
        super().__init__(path)


    def align_image(self, img: BaseImage,tail_elimination: bool = False) -> 'BaseImage':
        wys = img.data.shape[0]
        szer = img.data.shape[1]
        wart = img.data
        match wart.ndim:

            case 2:
                match tail_elimination:
                    case False:
                        min = np.amin(wart)
                        max = np.amax(wart)
                    case True:
                        hist = Histogram(wart).to_cumulated().valuesOfHistogram
                        minH = np.quantile(hist, 0.05)
                        maxH = np.quantile(hist, 0.95)
                        min = np.where(hist <=minH)
                        max = np.where(hist >=maxH)
                        try:
                            min = min[0][min[0].size-1]
                        except:
                            min = min[0]
                        try:
                            max = max[0][0]
                        except:
                            max = max[0]
                resArr = (wart - min) * (255/(max-min))
                for i in range(wys):
                    for j in range(szer):
                        if(resArr[i,j]>255):
                            resArr[i,j] = 255
                res = BaseImage(None)
                res.data = resArr.astype(np.uint8)
                res.color_model = 4
                return res
            case 3:
                temp = ImageAligning
                res1 = BaseImage(None)
                res1.data = img.get_layer(0)
                res2 = BaseImage(None)
                res2.data = img.get_layer(1)
                res3 = BaseImage(None)
                res3.data = img.get_layer(2)
                resArr1 = temp.align_image(self,res1, tail_elimination)
                resArr2 = temp.align_image(self, res2, tail_elimination)
                resArr3 = temp.align_image(self, res3, tail_elimination)
                tp = np.arange(wys * szer * 3).reshape(wys, szer, 3)

                for i in range(wys):
                    for j in range(szer):
                        r = np.array([resArr1.data[i, j], resArr2.data[i, j], resArr3.data[i, j]])
                        tp[i, j] = r

                Res = BaseImage(None)
                Res.data = tp.astype(np.uint8)
                Res.color_model = 0
                return Res

class ImageFiltration:

    def conv_2d(image: BaseImage, kernel: np.ndarray, prefix: Optional[float] = None) -> BaseImage:
        wys = image.data.shape[0]
        szer = image.data.shape[1]
        wart = image.data
        match wart.ndim:
            case 2:
                res = np.arange(wys*szer).reshape(wys, szer)
                for i in range(wys):
                    for j in range(szer):
                        r1 = i - floor(kernel.shape[0] / 2)
                        r2 = i + floor(kernel.shape[0]/2)+1
                        c1 = j - floor(kernel.shape[0]/2)
                        c2 = j + floor(kernel.shape[0]/2)+1
                        img = image.data[r1:r2, c1:c2]
                        try:
                            img = img * kernel
                            if(prefix != None):
                                sum = prefix * np.sum(img)
                            else:
                                sum = np.sum(img)
                        except:
                            sum = image.data[i, j]
                        res[i, j] = sum
                Res = BaseImage(None)
                Res.data = res.astype(np.uint8)
                Res.color_model = 4
                return Res

            case 3:

                tp = np.arange(wys * szer * 3).reshape(wys, szer, 3)
                temp = ImageFiltration
                img = image
                res1 = BaseImage(None)
                res1.data = img.get_layer(0)
                res2 = BaseImage(None)
                res2.data = img.get_layer(1)
                res3 = BaseImage(None)
                res3.data = img.get_layer(2)
                resArr1 = temp.conv_2d(res1, kernel, prefix)
                resArr2 = temp.conv_2d(res2, kernel, prefix)
                resArr3 = temp.conv_2d(res3, kernel, prefix)

                for i in range(wys):
                    for j in range(szer):
                        r = np.array([resArr1.data[i, j], resArr2.data[i, j], resArr3.data[i, j]])
                        tp[i, j] = r

                Res = BaseImage(None)
                Res.data = tp.astype(np.uint8)
                Res.color_model = 0
                return Res
        pass

class Thresholding(BaseImage):
    def __init__(self, path: str):
        super().__init__(path)

    def threshold(self, value: int, bi: BaseImage = None) -> BaseImage:
        if(bi==None):
            wys = self.data.shape[0]
            szer = self.data.shape[1]
            wart = self.data
        else:
            wys = bi.data.shape[0]
            szer = bi.data.shape[1]
            wart = bi.data
        match wart.ndim:
            case 2:
                res = np.arange(wys * szer).reshape(wys, szer)
                for i in range(wys):
                    for j in range(szer):
                        if(wart[i,j]<value):
                            res[i,j] = 0
                        elif(wart[i,j]>=value):
                            res[i, j] = 255
                Res = BaseImage(None)
                Res.data = res.astype(np.uint8)
                Res.color_model = 4
                return Res
            case 3:
                res = np.arange(wys * szer * 3).reshape(wys, szer,3)
                for i in range(wys):
                    for j in range(szer):
                        for k in range(3):
                            if (wart[i, j, k] < value):
                                res[i, j, k] = 0
                            elif (wart[i, j, k] >= value):
                                res[i, j, k] = 255
                Res = BaseImage(None)
                Res.data = res.astype(np.uint8)
                Res.color_model = 4
                return Res
        pass

class Opencv(BaseImage):

    def __init__(self, path: str):
        super().__init__(path)

    def toGrey(self) -> BaseImage:
        img_grayscale = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        Res = BaseImage(None)
        Res.data = img_grayscale.astype(np.uint8)
        return Res

    def fromAlpha(self) -> BaseImage:
        img_with_alpha = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
        Res = BaseImage(None)
        Res.data = img_with_alpha.astype(np.uint8)
        return Res

    def toRGB(self) -> BaseImage:
        img_rgb = cv2.cvtColor(self.path, cv2.COLOR_BGR2RGB)
        Res = BaseImage(None)
        Res.data = img_rgb.astype(np.uint8)
        return Res

    def threshold(self) -> BaseImage:
        lena_gray = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        _, thresh_otsu = cv2.threshold(
            lena_gray,
            thresh=0,
            maxval=255,
            type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        Res = BaseImage(None)
        Res.data = lena_gray.astype(np.uint8)
        return Res

    def adaptive(self) -> BaseImage:
        lena_gray = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        th_adaptive = cv2.adaptiveThreshold(
            lena_gray,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=13,
            C=8
        )
        Res = BaseImage(None)
        Res.data = th_adaptive.astype(np.uint8)
        return Res

    def detectionCanny(self) -> BaseImage:
        lena_gray = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        canny_edges = cv2.Canny(
            lena_gray,
            16,
            40,
            3
        )
        Res = BaseImage(None)
        Res.data = canny_edges.astype(np.uint8)
        return Res

    def CLAHE(self) -> BaseImage:
        lake_color = cv2.imread(self.path, cv2.IMREAD_COLOR)
        lake_gray = cv2.cvtColor(lake_color, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(4, 4)
        )
        equalized_lake_gray = clahe.apply(lake_gray)
        Res = BaseImage(None)
        Res.data = equalized_lake_gray.astype(np.uint8)
        return Res

    def korHisKol(self) -> BaseImage:
        lake_color = cv2.imread(self.path, cv2.IMREAD_COLOR)
        lake_rgb = cv2.cvtColor(lake_color, cv2.COLOR_BGR2RGB)
        lake_lab = cv2.cvtColor(lake_color, cv2.COLOR_BGR2LAB)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lake_lab[..., 0] = clahe.apply(lake_lab[..., 0])
        lake_color_equalized = cv2.cvtColor(lake_lab, cv2.COLOR_LAB2RGB)
        Res = BaseImage(None)
        Res.data = lake_color_equalized.astype(np.uint8)
        return Res

    def straightLineDetection(self) -> BaseImage:
        lines_img = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        _, lines_thresh = cv2.threshold(
            lines_img,
            thresh=0,
            maxval=255,
            type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        Res = BaseImage(None)
        Res.data = lines_img.astype(np.uint8)
        return Res

    def edgesOfLines(self) -> BaseImage:
        lines_img = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        _, lines_thresh = cv2.threshold(
            lines_img,
            thresh=0,
            maxval=255,
            type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        lines_edges = cv2.Canny(lines_thresh, 20, 50, 3)
        lines = cv2.HoughLinesP(
            lines_edges,
            2,
            np.pi / 180,
            30
        )
        Res = BaseImage(None)
        Res.data = lines.astype(np.uint8)
        return Res

    def circles(self, d=2, mD=60, miR=20, maR=100) -> BaseImage:
        checkers_img = cv2.imread(self.path)
        checkers_gray = cv2.cvtColor(checkers_img, cv2.COLOR_BGR2GRAY)
        checkers_color = cv2.cvtColor(checkers_img, cv2.COLOR_BGR2RGB)
        circles = cv2.HoughCircles(
            checkers_gray,
            method=cv2.HOUGH_GRADIENT,
            dp=d,
            minDist=mD,
            minRadius=miR,
            maxRadius=maR
        )
        for (x, y, r) in circles.astype(int)[0]:
            cv2.circle(checkers_color, (x, y), r, (0, 255, 0), 4)
        Res = BaseImage(None)
        Res.data = checkers_color.astype(np.uint8)
        return Res

# sepia and grey
b = BaseImage('lena.jpg')
g=GrayScaleTransform("lena.jpg")
g.to_sepia(w=20).save_img("obraz.jpg")
g.to_gray().show_img()

# histogram
bo = BaseImage("obraz.jpg")
b=ImageComparison("lena.jpg")
# b.histogram().plot() #wyswietlenie histogramu
print(b.compare_to(bo,0)) #prównywanie
print(b.compare_to(bo,1))

# histogram equalization
b = BaseImage('lena.jpg')
b1=ImageAligning("lena.jpg")
b1.align_image(b, False).save_img('obraz.jpg') # wyrównanie histogramu
h = Histogram(b.data)
h.to_cumulated().plot() #wyswietlanie skumulowanego histogramu
h.plot() #wyświetlanie zwykłego histogramu

# filtration
b=BaseImage("lena.jpg")
f=ImageFiltration
temp =np.array([[1,2,1],[2,4,2],[1,2,1]])
f.conv_2d(b,temp,1/16).save_img("obraz.jpg")

# binarization
b=BaseImage("lena.jpg")
temp=Thresholding("lena.jpg")
g = GrayScaleTransform('lena.jpg')
temp.threshold(150, g.to_sepia(w=30)).save_img("obraz.jpg")

# OpenCV
oc=Opencv('lena.jpg')
oc.adaptive().show_img()