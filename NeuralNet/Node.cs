using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    //Class tượng trưng cho các node trong layer nắm giữ bias và hàm kích hoạt được sử dụng để quyết định đầu ra dựa vào đầu vào cho trước
    class Node
    {
        public List<Link> linksSau = new List<Link>();
        public List<Link> linksTruoc = new List<Link>();
        public double bias;
        double deltaBias = 0; 
        public double gtriSauSigmoid = 0; //giá trị activation
        public double gtriTruocSigmoid; //giá trị trước khi tính sigmoid
        public double dCda;
        public int y = 0;
        double doDoc;// độ dốc, dùng để điều chỉnh lại trọng số lúc lan truyền ngược

        public Node(double bias)
        {
            this.bias = bias;
        }

        public void KetNoiNodeTruoc(Link connection)
        {
            linksTruoc.Add(connection);
        }
        public void KetNoiNodeSau(Link connection)
        {
            linksSau.Add(connection);
        }

        public void SetgtriSauSigmoid(double gtriSauSigmoid)
        {
            this.gtriSauSigmoid = gtriSauSigmoid;
        }

        public void SetBias(double bias)
        {
            this.bias = bias;
        }

        public void GanDelta(double learnRate)
        {   
            //Gán DeltaBias của node
            doDoc = dCda * SigmoidD(gtriTruocSigmoid);
            deltaBias += (doDoc * (learnRate * -1));

            //Gán deltaWeight của mọi weight liền kề nó
            foreach (var link in linksSau)
            {
                link.SetDeltaWeight(doDoc, learnRate);
            }

        }

        //Gán giá trị và gtriTruocSigmoid của node
        public virtual void LanTruyenTien()
        {
            double tong = 0;
            foreach (var link in linksSau)
                tong += link.WeightedgtriSauSigmoid;
            gtriTruocSigmoid = (tong + bias);
            gtriSauSigmoid = Sigmoid(gtriTruocSigmoid);

        }

        //Hàm tính toán sigmoid
        public double Sigmoid(double x)
        {
            return (1.0 / (1.0 + Math.Exp(-x)));
        }

        //Hàm tính đạo hàm của sigmoid
        public double SigmoidD(double x)
        {
            double var = Sigmoid(x);
            return (var * (1.0 - var));
        }

        //Tính đạo hàm của giá trị liên quan đến hàm kích hoạt output layer
        public void CalculateOutputDCda()
        {
            dCda = 2.0 * (gtriSauSigmoid - y);
        }

        //Tính đạo hàm của giá trị liên quan đến hàm kích hoạt
        public void CalculateDCda()
        {
            double tong = 0.0;
            foreach (var link in linksTruoc)
            {
                tong += (link.GetWeight() * (link.nodeSau.doDoc));
            }
            dCda = tong;
        }
        
        //Gán giá trị mới cho Bias= tổng trung bình cộng của số lần lặp lại mà nó lan truyền ngược
        public void SetNewBias(int repetitions)
        {
            bias = (deltaBias/repetitions) + bias;
            foreach (var link in linksSau)
            {
                link.SetNewWeight(repetitions);
            }
            
            //gán lại deltaBias
            deltaBias = 0;
        }
    }


}
