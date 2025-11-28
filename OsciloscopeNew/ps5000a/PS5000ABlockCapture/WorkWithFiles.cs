using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace WorkWithFile
{
    class WorkWithFiles
    {
        static public void SaveFilter(string filename, double[] filter, double[] x_arg)
        {
            Complex[] buf = new Complex[x_arg.Length];
            for (int i = 0; i < x_arg.Length; i++)
            {
                buf[i] = new Complex(x_arg[i], filter[i]);
            }
            Save2File(filename, buf);
        }

        static public void Save2File(string filename, double[] data)
        {
            using (StreamWriter Writer = new StreamWriter(filename))
            {
                string specifier = "E05";
                CultureInfo culture = CultureInfo.CreateSpecificCulture("en-US");
                //Writer.WriteLine(data.Length);
                for (int i = 0; i < data.Length; i++)
                {
                    Writer.WriteLine(data[i].ToString(specifier, culture).Replace(',', '.'));
                }
                Writer.Flush();
                Writer.Close();
            }
        }
        static public void Save2File(string filename, double[] data, int step)
        {
            using (StreamWriter Writer = new StreamWriter(filename))
            {
                string specifier = "E05";
                CultureInfo culture = CultureInfo.CreateSpecificCulture("en-US");
                //Writer.WriteLine(data.Length);
                for (int i = 0; i < data.Length; i += step)
                {
                    Writer.WriteLine(data[i].ToString(specifier, culture).Replace(',', '.'));
                }
                Writer.Flush();
                Writer.Close();
            }
        }


        static public void Save2File(string filename, Complex[] data, int step)
        {
            using (StreamWriter Writer = new StreamWriter(filename))
            {
                //Writer.WriteLine(data.Length);
                for (int i = 0; i < data.Length; i += step)
                {
                    Writer.WriteLine(data[i].ToString().Replace(',', '.'));
                }
                Writer.Flush();
                Writer.Close();
            }
        }


        static public void Save2File(string filename, Complex[] data)
        {
            using (StreamWriter Writer = new StreamWriter(filename))
            {
                //Writer.WriteLine(data.Length);
                for (int i = 0; i < data.Length; i++)
                {
                    Writer.WriteLine(data[i].ToString().Replace(',', '.'));
                }
                Writer.Flush();
                Writer.Close();
            }
        }

        static public Complex[] LoadFilter(string filename)
        {
            return LoadFromFileC(filename);
        }
        static public Complex Str2Cmpl(string s)
        {
            int pos = s.IndexOf(".0.") + 2;
            string s1 = s.Substring(1, pos - 1).Replace('.', ',');
            string s2 = s.Substring(pos + 1, s.Length - pos - 3).Replace('.', ',');
            Complex r = new Complex(double.Parse(s1), double.Parse(s2));
            return r;

        }
        static public Complex[] LoadFromFileC(string filename)
        {
            using (StreamReader Reader = new StreamReader(filename))
            {
                int l = int.Parse(Reader.ReadLine());
                Complex[] A = new Complex[l];
                for (int i = 0; i < l; i++)
                {
                    A[i] = Str2Cmpl(Reader.ReadLine());
                    //   A[i] = Complex.Parse(Reader.ReadLine().Replace('.', ','));
                }
                Reader.Close();
                return A;
            }
        }
        static public async void Save2FileAsync(string filename, double[] data)
        {
            using (StreamWriter Writer = new StreamWriter(filename))
            {
                string specifier = "E05";
                CultureInfo culture = CultureInfo.CreateSpecificCulture("en-US");
                //Writer.WriteLine(data.Length);
                for (int i = 0; i < data.Length; i++)
                {
                    await Writer.WriteLineAsync(data[i].ToString(specifier, culture).Replace(',', '.'));
                }
                Writer.Flush();
                Writer.Close();
            }
        }

        static public async void Save2FileAsync(string filename, double[] data, int step)
        {

            using (StreamWriter Writer = new StreamWriter(filename))
            {
                string specifier = "E05";
                CultureInfo culture = CultureInfo.CreateSpecificCulture("en-US");
                //Writer.WriteLine(data.Length);
                for (int i = 0; i < data.Length; i += step)
                {
                    await Writer.WriteLineAsync(data[i].ToString(specifier, culture).Replace(',', '.'));
                }
                Writer.Flush();
                Writer.Close();
            }

        }

    }
}
