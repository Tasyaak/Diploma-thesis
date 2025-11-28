using System;
using System.IO;
using System.IO.Compression;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Windows;
using static System.Numerics.Complex;

using System.Numerics;
namespace PS5000A
{

    public class FileMFT
    {

        private int len;
        private double[] Data;
        private byte[] bufb;
        //   double: хранит число с плавающей точкой от ±5.0*10-324 до ±1.7*10308 и занимает 8 байта.Пр
        private string filename;
        private MemoryStream Mem;
        private double minfreq;
        private double real_dt;
        private int freqs;
        private double[] arrA;


        public double[] Transform(int fcount_min, double fmin, double real_DT)
        {

            double min_freq1 = 1.0 / (real_DT * len);
            double min_fstep = min_freq1;
            //Определение частототы до которой надо начитать фурье, наименьшей допустимой
            if (fmin > min_freq1)
                min_freq1 = fmin;
            //ну и число частот, сколько будем считать
            int fcount = (int)(min_freq1 / min_fstep)+1;
            if (fcount < fcount_min)
                fcount = fcount_min;
            double[] Integrals = new double[fcount];
            //====================================
            minfreq = min_freq1;

            //======================================
            Complex arg_ = new Complex(0.0, Math.PI / ((double)len));
            Complex fi0 = Complex.Exp(arg_);
            Complex[] V0 = new Complex[len];
            Complex[] VM = new Complex[len];
            Complex[] Vbuf = new Complex[len];
            for (int i = 0; i < len; i++)
            {
                V0[i] = 1.0;
            }
            V0[0] = 0.5;
            V0[len-1] = 0.5;

            VM[0] = 1.0;
            for (int i = 1; i < len; i++)
            {
                VM[i] = VM[i - 1] * fi0;
            }
            Vbuf = V0;
            for (int i = 1; i < fcount; i++)
            {
                Integrals[i] = 0;
                for (int j = 1; j < len; j++)
                {
                    Integrals[i] = Integrals[i]  +Data[j]* Vbuf[j].Real;
                    Vbuf[j] = Vbuf[j] * VM[j];
                }
                Integrals[i] = Integrals[i] * 2 / ((double)len);
            }
            return Integrals;
        }

        public void PackFurieSave(string filename)
        {
            double[] packed;
            real_dt = 16.0E-9;
            packed = Transform(100, 500000.0, real_dt);
            //теперь пишем файл
            FileInfo fileInf = new FileInfo(filename);
            if (fileInf.Exists)
            {
                fileInf.Delete();
            }
            FileStream file = new FileStream(filename, FileMode.CreateNew, FileAccess.Write); //открывает файл только на чтение 
            freqs = packed.Length;
            //заголовок
            bufb = BitConverter.GetBytes(freqs);
            file.Write(bufb, 0, 4);
            bufb = BitConverter.GetBytes(len);
            file.Write(bufb, 0, 4);
            bufb = BitConverter.GetBytes(minfreq);
            file.Write(bufb, 0, 8);
            bufb = BitConverter.GetBytes(real_dt);
            file.Write(bufb, 0, 8); 
            //тело
            for (int i = 0; i < freqs; i++)
            {
                bufb = BitConverter.GetBytes(packed[i]);
                file.Write(bufb, 0, 8);
            }
            file.Flush(true);
            file.Close();
            file.Dispose();
        }

        public void LoadUnpackFurie(string filename)
        {
            double[] packed;
            FileInfo fileInf = new FileInfo(filename);
            if (fileInf.Exists)
            {
                FileStream file = new FileStream(filename, FileMode.Open, FileAccess.Read); //открывает файл только на чтение 
                file.Seek(0,SeekOrigin.Begin);
                file.Read(bufb, 0, 4);
                freqs = BitConverter.ToInt32(bufb, 0);
                file.Read(bufb, 0, 4);
                len = BitConverter.ToInt32(bufb, 0);
                file.Read(bufb, 0, 8);
                minfreq = BitConverter.ToDouble(bufb, 0);
                file.Read(bufb, 0, 8);
                real_dt = BitConverter.ToDouble(bufb, 0);
                packed = new double[freqs];
                for (int i = 0; i < freqs; i++)
                {
                    file.Read(bufb, 0, 8);
                    packed[i] = BitConverter.ToDouble(bufb, 0); 
                }
                file.Close();




            }
        }


        public void SetSize(int l)
        {
            len = l;
            Data = new double[len];
        }
        public void SetVal(double v, int i)
        {
            Data[i] = v;
        }
        public FileMFT()
        { 
            bufb = new byte[8];
        }
        public void SetName(string name_)
        {
            filename = name_;
        }
        public void LoadMFT()
        {
            LoadFile();
            Mem.Seek(0, SeekOrigin.Begin);
            // unsafe
            //  {              }
            Mem.Read(bufb, 0, 4);
            len = BitConverter.ToInt32(bufb, 0);
            Data = new double[len];
            for (int i = 0; i < len; i++)
            {
                Mem.Read(bufb, 0, 8);
                Data[i] = BitConverter.ToDouble(bufb, 0); 
            }

        }
        public void SaveMFT()
        {
            if (Mem == null)
            {
                Mem = new MemoryStream();
            }
            Mem.Seek(0, SeekOrigin.Begin); 
            bufb = BitConverter.GetBytes(len);
            Mem.Write(bufb, 0, 4);
            for (int i = 0; i < len; i++)
            {
                bufb = BitConverter.GetBytes(Data[i]);
                Mem.Write(bufb, 0, 8);
            } 
            SaveFile();
        }
        public void SaveMFTZ()
        {
            if (Mem == null)
            {
                Mem = new MemoryStream();
            }
            Mem.Seek(0, SeekOrigin.Begin); 
            bufb = BitConverter.GetBytes(len);
            Mem.Write(bufb, 0, 4);
            for (int i = 0; i < len; i++)
            {
                bufb = BitConverter.GetBytes(Data[i]);
                Mem.Write(bufb, 0, 8);
            } 
            FileInfo fileInf = new FileInfo(filename);
            if (fileInf.Exists)
            {
                fileInf.Delete();
            }
            FileStream file = new FileStream(filename, FileMode.CreateNew, FileAccess.Write); //открывает файл только на чтение 
            GZipStream gzipStream = new GZipStream(file, CompressionLevel.Optimal);
            Mem.Seek(0, SeekOrigin.Begin);
            Byte[] buffer = new Byte[Mem.Length];
            int h;
            while ((h = Mem.Read(buffer, 0, 1)) > 0)
            {
                
                   gzipStream.Write(buffer, 0, h);
            }  
            file.Flush(true);
            file.Close();
            file.Dispose();
        }
        public void SaveMFTD()
        {
            if (Mem == null)
            {
                Mem = new MemoryStream();
            }
            Mem.Seek(0, SeekOrigin.Begin); 
            bufb = BitConverter.GetBytes(len);
            Mem.Write(bufb, 0, 4);
            for (int i = 0; i < len; i++)
            {
                bufb = BitConverter.GetBytes(Data[i]);
                Mem.Write(bufb, 0, 8);
            } 
            FileInfo fileInf = new FileInfo(filename);
            if (fileInf.Exists)
            {
                fileInf.Delete();
            }
            FileStream file = new FileStream(filename, FileMode.CreateNew, FileAccess.Write); //открывает файл только на чтение 
            DeflateStream gzipStream = new DeflateStream(file, CompressionLevel.Optimal);
            Mem.Seek(0, SeekOrigin.Begin);
            Byte[] buffer = new Byte[Mem.Length];
            int h;
            while ((h = Mem.Read(buffer, 0, 1)) > 0)
            {

                gzipStream.Write(buffer, 0, h);
            }  
            file.Flush(true);
            file.Close();
            file.Dispose();
        }
        public void LoadMFTZ()
        {
            FileInfo fileInf = new FileInfo(filename);
            if (fileInf.Exists)
            {
                if (Mem ==null)
                { 
                    Mem = new MemoryStream(); 
                }
                FileStream file = new FileStream(filename, FileMode.Open, FileAccess.Read); //открывает файл только на чтение
                using (GZipStream gzipStream = new GZipStream(file, CompressionMode.Decompress))
                {
                    Byte[] buffer = new Byte[file.Length];
                    int h;
                    while ((h = gzipStream.Read(buffer, 0, 1)) > 0)
                    {
                        Mem.Write(buffer, 0, h);
                    }
                    file.Close();
                    file.Dispose();
                } 
                Mem.Seek(0, SeekOrigin.Begin); 
                Mem.Read(bufb, 0, 4);
                len = BitConverter.ToInt32(bufb, 0);
                Data = new double[len];
                for (int i = 0; i < len; i++)
                {
                    Mem.Read(bufb, 0, 8);
                    Data[i] = BitConverter.ToDouble(bufb, 0); 
                }
            }
        }


        public void LoadMFTD()
        {
            FileInfo fileInf = new FileInfo(filename);
            if (fileInf.Exists)
            {
                if (Mem == null)
                {
                    Mem = new MemoryStream();
                }
                FileStream file = new FileStream(filename, FileMode.Open, FileAccess.Read); //открывает файл только на чтение
                using (DeflateStream gzipStream = new DeflateStream(file, CompressionMode.Decompress))
                {
                    Byte[] buffer = new Byte[file.Length];
                    int h;
                    while ((h = gzipStream.Read(buffer, 0, 1)) > 0)
                    {
                        Mem.Write(buffer, 0, h);
                    }
                    file.Close();
                    file.Dispose();
                } 
                Mem.Seek(0, SeekOrigin.Begin); 
                Mem.Read(bufb, 0, 4);
                len = BitConverter.ToInt32(bufb, 0);
                Data = new double[len];
                for (int i = 0; i < len; i++)
                {
                    Mem.Read(bufb, 0, 8);
                    Data[i] = BitConverter.ToDouble(bufb, 0);

                }
            }
        }

        public FileMFT(string name)
        {
            filename = name;
        }

        public void LoadFile()
        {
            FileInfo fileInf = new FileInfo(filename);
            if (fileInf.Exists)
            {
                FileStream file = new FileStream(filename, FileMode.Open, FileAccess.Read); //открывает файл только на чтение
                Mem = new MemoryStream();
                file.CopyTo(Mem);
                file.Close();
                file.Dispose();
            }
        }
        public void LoadFile(string name)
        {
            filename = name;
            LoadFile();
        }
        public void SaveFile()
        {
            FileInfo fileInf = new FileInfo(filename);
            if (fileInf.Exists)
            {
                fileInf.Delete();
            }
            FileStream file = new FileStream(filename, FileMode.CreateNew, FileAccess.Write); //открывает файл только на чтение 
            Mem.Seek(0, SeekOrigin.Begin);
            Mem.CopyTo(file);
            file.Flush(true);
            file.Close();
            file.Dispose();
        }
        public void SaveFile(string name)
        {
            filename = name;
            SaveFile();
        }
        public void LoadTXT()
        {
            string[] arStr = File.ReadAllLines(filename);
            len = arStr.Length;
            Data = new double[len];
            for (int i = 0; i < len; i++)
            {
                Data[i] = double.Parse(arStr[i].Replace('.', ','));
            }
        }
        public void SaveTXT()
        {
         FileInfo fileInf = new FileInfo(filename);
            if (fileInf.Exists)
            {
                fileInf.Delete();
            }
            string[] arStr = new string[len]; 
            for (int i = 0; i < len; i++)
            {
                arStr[i] = Data[i].ToString().Replace(',','.'); 
            }
            File.WriteAllLines(filename, arStr); 
        }

    }
}
