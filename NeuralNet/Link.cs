using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    //The connection between a node nodeTruoc one layer nodeSau the next
    //Class chịu trách nhiệm kết nối giữa các layer với nhau, chứa weight giúp cho việc tính nodeSauán ndoe khi lan truyền mạng
    class Link
    {
        public Node nodeTruoc;
        public Node nodeSau;
        double weight;
        double deltaWeight = 0; 

        public Link(Node nodeTruoc, Node nodeSau, double weight)
        {
            nodeTruoc.KetNoiNodeTruoc(this);
            this.nodeTruoc = nodeTruoc;
            nodeSau.KetNoiNodeSau(this);
            this.nodeSau = nodeSau;
            this.weight = weight;
        }

        public void SetWeight(double weight)
        {
            this.weight = weight;
        }

        public double GetWeight()
        {
            return  weight;
        }
        //Gán deltaWeight = bSlope * giá trị hàm activation của node* -tiLeHoc(Công thức trên wiki)
        public void SetDeltaWeight(double bSlope, double tiLeHoc)
        {
            deltaWeight = (bSlope * nodeTruoc.value) * (tiLeHoc * -1);
        }

        //This is the same as weight * activation of the nodeTruocNode
        public double WeightedValue
        {
            get {return (nodeTruoc.value * this.weight); }
        }
        //Gán lại giá trị weight=trung bình cộng của các weight
        public void SetNewWeight(int repetitions)
        {
            weight = (deltaWeight / repetitions) + weight;

            deltaWeight = 0;
        }

    }
}
