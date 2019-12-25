using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    //Contains the nodes that will define how the layer will impact layers in front, or the output if it's the last layer
    class Layer
    {
        public Node[] nodes;
        //numNodes is the ammount of nodes in the layer, rnd is the Random passed in to randomize the biases and weights
        public Layer(int numNodes, Random rnd)
        {
            double bias;
            nodes = new Node[numNodes];
            for (int i = 0; i < nodes.Length; i++)
            {
                bias = rnd.NextDouble() * 2 - 1;
                nodes[i] = new Node(bias);
            }
        }
    }
}
