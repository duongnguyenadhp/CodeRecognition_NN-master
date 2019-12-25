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
        public double value = 0; //giá trị activation
        public double gtriTruocSigmoid; //giá trị trước khi tính sigmoid
        public double dCda;
        public int y = 0;
        double bSlope;// độ dốc, dùng để điều chỉnh lại trọng số lúc lan truyền ngược

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

        public void SetValue(double value)
        {
            this.value = value;
        }

        public void SetBias(double bias)
        {
            this.bias = bias;
        }

        public void GanDelta(double learnRate)
        {   
            //Gán DeltaBias của node
            bSlope = dCda * SigmoidD(gtriTruocSigmoid);
            deltaBias += (bSlope * (learnRate * -1));

            //Gán deltaWeight của mọi weight liền kề nó
            foreach (var link in linksSau)
            {
                link.SetDeltaWeight(bSlope, learnRate);
            }

        }

        //Gán giá trị và gtriTruocSigmoid của node
        public virtual void LanTruyenTien()
        {
            double sum = 0;
            foreach (var link in linksSau)
                sum += link.WeightedValue;
            gtriTruocSigmoid = (sum + bias);
            value = Sigmoid(gtriTruocSigmoid);

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
            dCda = 2.0 * (value - y);
        }

        //Tính đạo hàm của giá trị liên quan đến hàm kích hoạt
        public void CalculateDCda()
        {
            double sum = 0.0;
            foreach (var link in linksTruoc)
            {
                sum += (link.GetWeight() * (link.nodeSau.bSlope));
            }
            dCda = sum;
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
