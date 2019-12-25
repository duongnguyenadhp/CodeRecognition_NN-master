using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace NeuralNet
{
    public partial class Form1 : Form
    {
        int count = 49999;//50k lần đào tạo 
        int soLanDoanDung = 0;
        int tocDoChuyenAnh = 1000; //Thời gian để chuyển ảnh
        NeuralNetwork nn = new NeuralNetwork();
        public Form1()
        {
            InitializeComponent();
        }
        //chuyển đổi mảng ByteArray sang Image
        public Image byteArrayToImage(byte[] byteArrayIn)
        {
            int index = 0;
            Bitmap bmp = new Bitmap(28,28);

            for (int i = 0; i < bmp.Height ; i++)
            {
                int gray = 40;
                int background = 105;
                for (int j = 0; j < bmp.Width; j++, index++)
                {
                    Color c = Color.FromArgb(byteArrayIn[index], byteArrayIn[index], byteArrayIn[index]);
                    if (c.R > 100)
                    {
                        c = Color.FromArgb(gray, gray, gray);
                    }
                    else if (c.R < 50)
                    {
                        c = Color.FromArgb(background, background, background);
                    }
                    bmp.SetPixel(j, i, c);
                }
            }

            return (Image)bmp;
        }
        private void timer1_Tick(object sender, EventArgs e)
        {
            CapNhatInput();            
        }

        //Cập nhật image box và label 
        private void CapNhatInput()
        {
            if (count < nn.mnistTrain.labels.Length)
            {
                pictureBox1.Image = byteArrayToImage(nn.mnistTrain.images[count]);
                label3.Text = "" + nn.mnistTrain.labels[count];
                nn.GanDuLieuInput(nn.mnistTrain.images[count], count);
                nn.LanTruyenTienNeural();
                label4.Text = "" + nn.LayOutput();

                if (nn.mnistTrain.labels[count] == nn.LayOutput())
                {
                    soLanDoanDung++;
                }
                label6.Text = $"{soLanDoanDung}/{count - 49998}";
                count++;
            }

            if (tocDoChuyenAnh != 1)
            {
                tocDoChuyenAnh--;
                timer1.Interval = tocDoChuyenAnh;
            }

        }
        public void InitTimer()
        {
            timer1 = new System.Windows.Forms.Timer();
            timer1.Tick += new EventHandler(timer1_Tick);
            timer1.Interval = tocDoChuyenAnh; 
            timer1.Start();
        }

        private void trainToolStripMenuItem_Click(object sender, EventArgs e)
        {
                Thread trainningThread = new Thread(() => nn.TrainNN());
                trainningThread.Start();
        }
        private void testToolStripMenuItem_Click(object sender, EventArgs e)
        {
            InitTimer();
        }
    }
}
