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
            CvInvoke.Imshow("Roi", roi);
            var roiHsv = new Mat();
            CvInvoke.CvtColor(roi, roiHsv, ColorConversion.Bgr2Hsv);
            var histogram = new Mat();
            Console.WriteLine(histogram.ToString());
            CvInvoke.CalcHist(new VectorOfMat(new Mat[] { roiHsv }), new int[] { 0 }, null, histogram, new int[] { 180 }, new float[] { 0, 180 }, false);
            CvInvoke.Normalize(histogram, histogram, 0, 255, NormType.MinMax);

            var nextFrame = new Mat();
            var nextFrameHsv = new Mat();
            var mask = new Mat();
            var trackingWindow = new Rectangle(x, y, width, height);
            while (true)
            {
                video.Read(nextFrame);
                CvInvoke.CvtColor(nextFrame, nextFrameHsv, ColorConversion.Bgr2Hsv);
                CvInvoke.CalcBackProject(new VectorOfMat(new Mat[] { nextFrameHsv }), new int[] { 0 }, histogram, mask, new float[] { 0, 180 }, 1);
                CvInvoke.Imshow("mask", mask);
                CvInvoke.MeanShift(mask, ref trackingWindow, new MCvTermCriteria(10, 1));
                CvInvoke.Rectangle(nextFrame, trackingWindow, new MCvScalar(0, 255, 0), 2);
                CvInvoke.Imshow("nextFrame", nextFrame);
                CvInvoke.WaitKey(60);
            }
        }
    }
}
