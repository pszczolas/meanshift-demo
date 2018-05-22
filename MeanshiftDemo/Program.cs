using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System.Drawing;
using Emgu.CV.Util;

namespace MeanshiftDemo
{
    class Program
    {
        static void Main(string[] args) => RunMeanshiftDemo();

        static void RunMeanshiftDemo()
        {
            VideoCapture video = new VideoCapture("mleko.mp4");  // mleko.mp4
            //VideoCapture video = new VideoCapture("mouthwash.avi");  // mouthwash.avi
            var firstFrame = new Mat();
            video.Read(firstFrame);
            int x = 290, y = 230, width = 100, height = 15; // mleko.mp4
            //int x = 300, y = 305, width = 100, height = 115; // mouthwash.avi
            var roi = new Mat(firstFrame, new Rectangle(x, y, width, height));
            ShowHueEmphasizedImage(roi);
            
            CvInvoke.Imshow("Roi", roi);
            var roiHsv = new Mat();
            CvInvoke.CvtColor(roi, roiHsv, ColorConversion.Bgr2Hsv);
            var histogram = new Mat();
            CvInvoke.CalcHist(new VectorOfMat(new Mat[] { roiHsv }), new int[] { 0 }, null, histogram, new int[] { 180 }, new float[] { 0, 180 }, false);
            CvInvoke.Normalize(histogram, histogram, 0, 255, NormType.MinMax);

            show2DHueHistogram(histogram);

            var nextFrame = new Mat();
            var nextFrameHsv = new Mat();
            var mask = new Mat();
            var trackingWindow = new Rectangle(x, y, width, height);
            while (true)
            {
                video.Read(nextFrame);
                if (nextFrame.IsEmpty)
                    break;
                CvInvoke.CvtColor(nextFrame, nextFrameHsv, ColorConversion.Bgr2Hsv);
                CvInvoke.CalcBackProject(new VectorOfMat(new Mat[] { nextFrameHsv }), new int[] { 0 }, histogram, mask, new float[] { 0, 180 }, 1);
                CvInvoke.Imshow("mask", mask);
                CvInvoke.MeanShift(mask, ref trackingWindow, new MCvTermCriteria(10, 1));
                CvInvoke.Rectangle(nextFrame, trackingWindow, new MCvScalar(0, 255, 0), 2);
                CvInvoke.Imshow("nextFrame", nextFrame);
                CvInvoke.WaitKey(60);
            }
            Console.WriteLine("Koniec filmu.");
            CvInvoke.WaitKey();
        }

        private static void ShowHueEmphasizedImage(Mat bgrMat)
        {
            var roiImg = bgrMat.ToImage<Bgr, byte>();
            var roiImgHsv = new Image<Hsv, byte>(roiImg.Width, roiImg.Height);
            CvInvoke.CvtColor(roiImg, roiImgHsv, ColorConversion.Bgr2Hsv);
            for (int h = 0; h < roiImg.Height; h++)
            {
                for (int w = 0; w < roiImg.Width; w++)
                {
                    roiImgHsv[h, w] = new Hsv(roiImgHsv[h, w].Hue, 255, 255);
                }
            }
            CvInvoke.CvtColor(roiImgHsv, roiImg, ColorConversion.Hsv2Bgr);
            CvInvoke.Imshow("emphasized hue", roiImg);
        }

        private static void show2DHueHistogram(Mat histogram)
        {
            var histogram1D = histogram.ToImage<Hsv, float>();
            var histogram2D = new Image<Hsv, byte>(180, 256, new Hsv());
            var hueScale = CreateHueScale();
            for (int h = 0; h < histogram1D.Height; h++)
            {
                for (int i = 0; i < histogram1D.Data[h, 0, 2]; i++)
                {
                    var tmpValue = hueScale[0, h];
                    histogram2D[i, h] = tmpValue;
                }
            }
            CvInvoke.CvtColor(histogram2D, histogram2D, ColorConversion.Hsv2Bgr);
            CvInvoke.Imshow("histogram", histogram2D);
        }

        private static Image<Hsv, byte> CreateHueScale()
        {
            const int maxHue = 180;
            const int maxSaturation = 255;
            const int maxValue = 255;
            var hueScale = new Image<Hsv, byte>(maxHue, 5, new Hsv());
            for (int i = 0; i < hueScale.Width; i++)
            {
                hueScale[0, i] = new Hsv(i, maxSaturation, maxValue);
            }

            return hueScale;
        }
    }
}
