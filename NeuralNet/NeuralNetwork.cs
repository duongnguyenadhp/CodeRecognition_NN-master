using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace NeuralNet
{
    class NeuralNetwork
    {
        Layer tangInput;
        Layer tangAn1;
        Layer tangAn2;
        Layer tangOutput;
        double tiLeHoc = 3.0;
        //Số lần lặp lại trước khi cập nhật weight và bias
        int lapLai = 10; //Số lần lặp lại trước khi gán weight mới và bias
        // 28px x 28px=748
        const int KICH_THUOC_ANH = 784;
        const int KICH_THUOC_OUTPUT = 10;

        Random rnd = new Random();

        //Khởi tạo cơ sở dữ liệu mnisTrain chứa các bộ dữ liệu để training và test
        public MNISTRead mnistTrain = new MNISTRead();

        int index = 0;
        //Tạo các layer, cài đặt đầu vào, kết nối các layer với nhau, lan truyền tiến, lan truyền ngược
        public NeuralNetwork()
        {
            tangInput = new Layer(KICH_THUOC_ANH, rnd);
            tangAn1 = new Layer(16, rnd);
            tangAn2 = new Layer(16, rnd);
            tangOutput = new Layer(KICH_THUOC_OUTPUT, rnd);
            new Random().TronDuLieu(mnistTrain.images, mnistTrain.labels);
            GanDuLieuInput(mnistTrain.images[index], index);
            KetNoiTang(tangInput, tangAn1);
            KetNoiTang(tangAn1, tangAn2);
            KetNoiTang(tangAn2, tangOutput);
        }

        public void GanDuLieuInput(byte[] byteArrayIn, int index)
        {
            this.index = index;
            for (int i = 0; i < tangInput.nodes.Length; i++)
            {
                tangInput.nodes[i].gtriSauSigmoid = byteArrayIn[i] / 255.0;
            }
        }
        // Trả về đầu ra của mạng neural và gán giá trị ys của các node trong tangOutput
        public int LayOutput()
        {
            double max = tangOutput.nodes[0].gtriSauSigmoid;
            int maxOutput = 0;
            for (int i = 1; i < tangOutput.nodes.Length; i++)
            {
                if (tangOutput.nodes[i].gtriSauSigmoid > max)
                {
                    max = tangOutput.nodes[i].gtriSauSigmoid;
                    maxOutput = i;
                }
            }
            return maxOutput;
        }
        //Gán giá trị y's của các node trong tangOutput
        public void GanYsOutput()
        {
            //Gán toàn bộ ys của node = 0
            for (int i = 0; i < tangOutput.nodes.Length; i++)
            {
                tangOutput.nodes[i].y = 0;
            }

            //Gán giá trị mong muốn = 1, các giá trị còn lại = 0
            int k = mnistTrain.labels[index];
            tangOutput.nodes[k].y = 1;

        }
        //Lan truyền tiến toàn bộ các lớp trong mạng neural, ngoại trừ layer input
        public void LanTruyenTienNeural()
        {
            for (int i = 0; i < tangAn1.nodes.Length; i++)
            {
                tangAn1.nodes[i].LanTruyenTien();
            }
            for (int i = 0; i < tangAn2.nodes.Length; i++)
            {
                tangAn2.nodes[i].LanTruyenTien();
            }
            for (int i = 0; i < tangOutput.nodes.Length; i++)
            {
                tangOutput.nodes[i].LanTruyenTien();
            }
        }
        //Lan truyền ngược
        public void LanTruyenNguoc(Layer layer)
        {
            //Nếu layer cần lan truyền là ouput layer
            if (layer == tangOutput)
            {

                //Tính đạo hàm của giá trị hàm kích hoạt của mọi node
                for (int i = 0; i < layer.nodes.Length; i++)
                {
                    layer.nodes[i].CalculateOutputDCda();
                    layer.nodes[i].GanDelta(tiLeHoc);
                }

            }
            else
            {
                for (int i = 0; i < layer.nodes.Length; i++)
                {
                    layer.nodes[i].CalculateDCda();
                    layer.nodes[i].GanDelta(tiLeHoc);
                }
            }

        }

        //Kết nối layer trước và layer sau
        void KetNoiTang(Layer fromLayer, Layer toLayer)
        {
            for (int i = 0; i < toLayer.nodes.Length; i++)
            {
                for (int j = 0; j < fromLayer.nodes.Length; j++)
                {
                    new Link(fromLayer.nodes[j], toLayer.nodes[i], (rnd.NextDouble() * 2) - 1);
                }
            }
        }

        //Gán giá trị delta mới cho layer
        void GanDelta(Layer layer, int repetitions)
        {
            for (int i = 0; i < layer.nodes.Length; i++)
            {
                layer.nodes[i].SetNewBias(repetitions);
            }

        }

        //Hàm huấn luyện mạng neural
        public void TrainNN()
        {
            new Random().TronDuLieu(mnistTrain.images, mnistTrain.labels);
            //Số lần lặp lại toàn bộ tiến trình = mnistTrain.labels.Length / lapLai / 6 * 5
            int totalRepetitions = 5000;
            for (int j = 0; j < totalRepetitions; j++)
            {
                for (int i = 0; i < lapLai; i++)
                {
                    GanDuLieuInput(mnistTrain.images[index], index);
                    LanTruyenTienNeural();
                    GanYsOutput();
                    LanTruyenNguoc(tangOutput);
                    LanTruyenNguoc(tangAn2);
                    LanTruyenNguoc(tangAn1);
                    index++;
                }
                GanDelta(tangOutput, lapLai);
                GanDelta(tangAn2, lapLai);
                GanDelta(tangAn1, lapLai);
            }
            //Gán lại chỉ số
            index = 0;
            MessageBox.Show("Hoàn thành training!");
        }

    }

    static class RandomExtensions
    {
        //Trộn dữ liệu sử dụng hàm random
        public static void TronDuLieu<T>(this Random rng, T[][] images, T[] labels)
        {
            int n = images.Length;
            while (n > 1)
            {
                int k = rng.Next(n--);
                T[] temp = images[n];
                images[n] = images[k];
                images[k] = temp;
                T temp2 = labels[n];
                labels[n] = labels[k];
                labels[k] = temp2;
            }
        }
    }
}
