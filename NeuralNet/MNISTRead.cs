using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    //Đọc dữ liệu từ database có sẵn phục vụ cho việc trainning
    class MNISTRead
    {
        public byte[][] images;
        public byte[] labels;
        public MNISTRead()
        {
            string imagesPath = "..\\..\\Data\\train-images.idx3-ubyte"; //Path to Images
            string labelsPath = "..\\..\\Data\\train-labels.idx1-ubyte"; //Path to labels
            using (BinaryReader brImages = new BinaryReader(new FileStream(imagesPath, FileMode.Open)), brLabels = new BinaryReader(new FileStream(labelsPath, FileMode.Open)))
            {
                int magic1 = brImages.ReadInt32Endian();
                if (magic1 != 2051)
                    throw new Exception($"Invalid magic number {magic1}!");
                int numImages = brImages.ReadInt32Endian();
                int numRows = brImages.ReadInt32Endian();
                int numCols = brImages.ReadInt32Endian();
                int magic2 = brLabels.ReadInt32Endian();
                if (magic2 != 2049)
                    throw new Exception($"Invalid magic number {magic2}!");
                int numLabels = brLabels.ReadInt32Endian();
                if (numLabels != numImages)
                    throw new Exception($"Number of labels ({numLabels}) does not equal number of images ({numImages})");

                images = new byte[numImages][];
                labels = new byte[numLabels];
                int dimensions = numRows * numCols;
                for (int i = 0; i < numImages; i++)
                {
                    images[i] = brImages.ReadBytes(dimensions);
                    labels[i] = brLabels.ReadByte();
                }
            }
        }
        
        

    }
}
