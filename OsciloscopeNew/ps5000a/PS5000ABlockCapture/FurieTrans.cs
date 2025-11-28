using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using AllConstants;

namespace AllFurieTrans
{
    class FurieTrans
    {
        static public Complex[] FurieTransf(double[] data, double dt, double t0, double f0, double df, int nf)
        {
            double w0 = 2 * Constants.M_PI * f0;
            double dw = 2 * Constants.M_PI * df;
            double mult = Math.Sqrt(1.0 / 2.0 / Constants.M_PI);
            Complex[] result = new Complex[nf];
            double w = w0;
            int l = data.Length;
            for (int j = 0; j < nf; j++)
            {
                Complex base_exp = Complex.Exp(-1 * w * t0 * Complex.ImaginaryOne);
                Complex mult_exp = Complex.Exp(-1 * w * dt * Complex.ImaginaryOne);
                Complex _exp = base_exp * mult_exp;
                result[j] = (data[0] * base_exp +
                data[l - 1] * Complex.Exp(-1 * w * (t0 + (double)(l - 1) * dt) * Complex.ImaginaryOne)) / 2.0;
                for (int k = 1; k < l - 1; k++)
                {
                    result[j] = result[j] + data[k] * _exp;
                    _exp = _exp * mult_exp;
                }
                result[j] = dt * result[j] * mult;
                w = w + dw;
            }
            return result;
        }
        static public double[] FuncMult(double[] f1, double f1dx, double f1x0, Complex[] filter)
        {
            double[] result = new double[f1.Length];
            double mult = 0;
            int j = 0;
            for (int i = 0; i < f1.Length; i++)
            {
                double x = f1x0 + f1dx * (double)i;
                if (x < filter[0].Real)
                {
                    mult = filter[0].Imaginary;
                }
                else
                {
                    if (x > filter[filter.Length - 1].Real)
                    {
                        mult = filter[filter.Length - 1].Imaginary;
                    }

                    else
                    {
                        while (filter[j + 1].Real < x)
                        {
                            j++;
                        }

                        double a = filter[j].Imaginary;
                        double b = (filter[j + 1].Imaginary - filter[j].Imaginary) / (filter[j + 1].Real - filter[j].Real);
                        double x_ = x - filter[j].Real;
                        mult = a + b * x_;


                        //============================================================================
                        //где нарушена логика, надо переписать
                        //============================================================================

                        //while ((filter[j].Real < x) && (j < (filter.Length - 1)))
                        //{
                        //    j++;
                        //}
                        //if (j == (filter.Length - 1))
                        //{
                        //    mult = filter[j].Real;
                        //}
                        //else
                        //{
                        //    double b = (filter[j + 1].Imaginary - filter[j].Imaginary) / (filter[j + 1].Real - filter[j].Real);
                        //    double a = filter[j].Imaginary;
                        //    double x_ = x - filter[j].Real;
                        //    mult = a + b * x_;
                        //}
                    }
                }
                result[i] = mult * f1[i];
            }
            return result;
        }
        static public Complex[] FurieTransfReverse(Complex[] data, double dt, double t0, int nt, double f0, double df)
        {
            double w0 = 2 * Constants.M_PI * f0;
            double dw = 2 * Constants.M_PI * df;
            double mult = Math.Sqrt(1.0 / (2.0 * Constants.M_PI));
            Complex[] result = new Complex[nt];
            double t = t0;
            int l = data.Length;
            for (int j = 0; j < nt; j++)
            {
                Complex base_exp = Complex.Exp(t * w0 * Complex.ImaginaryOne);
                Complex mult_exp = Complex.Exp(t * dw * Complex.ImaginaryOne);
                Complex _exp = base_exp * mult_exp;
                result[j] = (data[0] * base_exp +
                data[l - 1] * Complex.Exp(t * (w0 + (double)(l - 1) * dw)
                * Complex.ImaginaryOne)) / 2.0;
                for (int k = 1; k < l - 1; k++)
                {
                    result[j] = result[j] + data[k] * _exp;
                    _exp = _exp * mult_exp;
                }
                result[j] = dw * result[j] * mult;
                t = t + dt;
            }
            return result;
        }
        static public Complex[] FuncMult(Complex[] f1, double f1dx, double f1x0, Complex[] filter)
        {
            Complex[] result = new Complex[f1.Length];
            double mult = 0;
            int j = 0;
            for (int i = 0; i < f1.Length; i++)
            {
                double x = f1x0 + f1dx * (double)i;
                if (x < filter[0].Real)
                {
                    mult = filter[0].Imaginary;
                }
                else
                {
                    if (x > filter[filter.Length - 1].Real)
                    {
                        mult = filter[filter.Length - 1].Imaginary;
                    }

                    else
                    {
                        while (filter[j + 1].Real < x)
                        {
                            j++;
                        }

                        double a = filter[j].Imaginary;
                        double b = (filter[j + 1].Imaginary - filter[j].Imaginary) / (filter[j + 1].Real - filter[j].Real);
                        double x_ = x - filter[j].Real;
                        mult = a + b * x_;
                    }
                }
                result[i] = mult * f1[i];
            }

            return result;
        }
    }
}
