using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Ports;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace PS5000A
{
    public class Switch
    {
        public const int E_OK = 0;
        public const int E_TIMEOUT = 1;
        public const int E_CONNECTION = 2;
        public const int E_CONNECTION_LOST = 3;
        public const int E_TRANSMISSION_FAIL = 4;
        public SerialPort port;
        public byte[] Kirill_DATA;
        public int RelayLayers;
        private byte MASK_IN0 = Convert.ToByte("00001011", 2);
        private byte MASK_IN1 = Convert.ToByte("11010000", 2);
        private byte MASK_OFF0 = Convert.ToByte("11110000", 2);
        private byte MASK_OFF1 = Convert.ToByte("00001111", 2);
        private byte MASK_OUT0 = Convert.ToByte("00000111", 2);
        private byte MASK_OUT1 = Convert.ToByte("11100000", 2);
        // реле нумеруютчся от 0 до 2*N-1

        ~Switch()
        {
            Protocol_V2d0_disconnect();

            Console.WriteLine("Finalizing object");
        }

        public void OFFRelay(int num)
        {
            byte mask = MASK_OFF1;
            if ((num % 2) == 0)
            {
                mask = MASK_OFF0;
            }
            Kirill_DATA[num / 2] = (byte)(mask & Kirill_DATA[num / 2]);
        }
        public void INRelay(int num)
        {
            byte mask = MASK_IN1;
            if ((num % 2) == 0)
            {
                mask = MASK_IN0;
            }
            Kirill_DATA[num / 2] = (byte)(mask | Kirill_DATA[num / 2]);
        }
        public void OUTRelay(int num)
        {
            byte mask = MASK_OUT1;
            if ((num % 2) == 0)
            {
                mask = MASK_OUT0;
            }
            Kirill_DATA[num / 2] = (byte)(mask | Kirill_DATA[num / 2]);
        }
        public void NOT_INRelay(int num)
        {
            byte mask = MASK_IN1;
            if ((num % 2) == 0)
            {
                mask = MASK_IN0;
            }
            Kirill_DATA[num / 2] = (byte)((~mask) & Kirill_DATA[num / 2]);
        }
        public void NOT_OUTRelay(int num)
        {
            byte mask = MASK_OUT1;
            if ((num % 2) == 0)
            {
                mask = MASK_OUT0;
            }
            Kirill_DATA[num / 2] = (byte)((~mask) & Kirill_DATA[num / 2]);
        }
        public void sendKIRILL()
        {
            port.Write(Kirill_DATA, 0, RelayLayers);
            port.BaseStream.Flush();
        }
        public void InitKirill(int relaylayers, int portnum)
        {
            RelayLayers = relaylayers;
            Kirill_DATA = new byte[relaylayers];
            for (int i = 0; i < relaylayers; i++)
            {
                Kirill_DATA[i] = 0;
            }
            OpenPort(portnum);

        }
        public string Receive_str()
        {
            if (port.BytesToRead > 0)
            {
                return port.ReadLine();
            }
            else
            {
                return "";
            }
        }
        //public void OpenPort()
        //{
        //    // получаем список доступных портов 
        //    string[] ports = SerialPort.GetPortNames();
        //    Console.WriteLine("Выберите порт:");
        //    // выводим список портов
        //    for (int i = 0; i < ports.Length; i++)
        //    {
        //        Console.WriteLine("[" + i.ToString() + "] " + ports[i].ToString());
        //    }
        //    port = new SerialPort();
        //    // читаем номер из консоли
        //    string n = Console.ReadLine();
        //    int num = int.Parse(n);
        //    try
        //    {
        //        // настройки порта
        //        port.PortName = ports[num];
        //        port.BaudRate = 57600;
        //        port.DataBits = 8;
        //        port.Parity = System.IO.Ports.Parity.None;
        //        port.StopBits = System.IO.Ports.StopBits.One;
        //        port.ReadTimeout = 3000;
        //        port.WriteTimeout = 3000;
        //        port.Open();
        //    }
        //    catch (Exception e)
        //    {
        //        Console.WriteLine("ERROR: невозможно открыть порт:" + e.ToString());
        //        return;
        //    }
        //}

        public void OpenPort(int num)
        {
            // получаем список доступных портов 
            string[] ports = SerialPort.GetPortNames();
            try
            {
                port = new SerialPort
                {
                    // настройки порта
                    PortName = ports[num],
                    BaudRate = 19200,
                    DataBits = 8,
                    Parity = System.IO.Ports.Parity.Even,
                    StopBits = System.IO.Ports.StopBits.Two,
                    ReadTimeout = 3000,
                    WriteTimeout = 3000
                };
                port.Open();
            }
            catch (Exception e)
            {
                Console.WriteLine("ERROR: невозможно открыть порт:" + e.ToString());
                return;
            }
        }
        public void OpenPort(string name, int bod = 19200)
        {

            // получаем список доступных портов  
            //try
            //{
            //    port = new SerialPort
            //    {
            //        // настройки порта
            //        PortName = name,
            //        BaudRate = bod,
            //        DataBits = 8,
            //        Parity = System.IO.Ports.Parity.Even,
            //        StopBits = System.IO.Ports.StopBits.Two,
            //        ReadTimeout = 3000,
            //        WriteTimeout = 3000
            //    };
            //    port.Open();
            //}  
            //catch (Exception e)
            //{
            //    Console.WriteLine("ERROR: невозможно открыть порт:" + e.ToString());
            //    return;
            //}



            // получаем список доступных портов  
            try
            {
                port = new SerialPort
                {
                    // настройки порта
                    PortName = name,
                    BaudRate = bod,
                    DataBits = 8,
                    Parity = System.IO.Ports.Parity.None,
                    StopBits = System.IO.Ports.StopBits.One,
                    ReadTimeout = 1000,
                    WriteTimeout = 1000
                };
                port.Open();
            }
            catch (Exception e)
            {
                Console.WriteLine("ERROR: невозможно открыть порт:" + e.ToString());
                return;
            }

        }

        public void SendCmd(int status /*старшая цифра - команда */, int addr /*младшая цифра - адресс */)
        {
            port.Write(status.ToString() + addr.ToString());
            port.BaseStream.Flush();
        }
        //отправка команды ввиде строки
        public void SendCmd(string s)
        {
            port.Write(s);
            port.BaseStream.Flush();
        }
        public void SetOut(int addr)
        {
            SendCmd(0, addr);
        }
        public void SetIn(int addr)
        {
            SendCmd(1, addr);
        }
        //отправка команды из консоли
        // старшая цифра - команда младшая цифра - адресс 
        public void SendCmd2()
        {
            port.Write(Console.ReadLine());
        }
        public void ClosePort_()
        {
            if (port.IsOpen)
            {
                port.Close();
            }
        }

        public string GetAccepted()
        {
            if (port.IsOpen)
            {
                if (port.BytesToRead > 0)
                {
                    return port.ReadLine();
                }
            }
            return "";
        }
        public string GetAcceptedKiril()
        {
            if (port.IsOpen)
            {
                if (port.BytesToRead > 0)
                {
                    return port.ReadExisting();
                }
            }
            return "";
        }

        public string ReadALL()
        {
            string txt1 = "";

            if (port.IsOpen)
            {
                if (port.BytesToRead != 0)
                {
                    txt1 = GetAcceptedKiril();
                    port.BaseStream.Flush();

                    while (port.BytesToRead > 0)
                    {

                        txt1 += GetAcceptedKiril();
                        port.BaseStream.Flush();
                    }
                }
            }
            return txt1;
        }

        public void config_sens_count_on_device(byte count)
        {
            if (port.IsOpen)
            {
                byte[] temp = { count };
                port.Write(temp, 0, 1);
                port.BaseStream.Flush();
            }
        }

        public string Protocol_V2d0_connect(byte count)
        {
            string X = ReadALL();
            Thread.Sleep(1000);
            config_sens_count_on_device(count);
            Thread.Sleep(1000);
            X += ReadALL();
            return X;
        }


        public string Protocol_V2d0_send()
        {
            bool succesful_transmit = false;

            string txt1 = ReadALL();
            Thread.Sleep(100);
            txt1 += ReadALL();

            while (!succesful_transmit)
            {

                sendKIRILL();
                byte host_crc = crc8(Kirill_DATA, RelayLayers);
                while (port.BytesToRead == 0)
                { Thread.Sleep(10); }
                string L1 = port.ReadLine();
                while (port.BytesToRead == 0)
                { Thread.Sleep(10); }
                string L2 = port.ReadLine();

                byte device_received = byte.Parse(L1);
                byte device_crc8 = byte.Parse(L2);
                bool size_check = RelayLayers == device_received;
                bool crc_check = device_crc8 == host_crc;
                  succesful_transmit = size_check && crc_check;
                 
                txt1 += "SIZE_"+L1 +"_CRC8_"+ L2 + "\n";
                txt1 += ReadALL(); 
                txt1 += ReadALL();
                if (succesful_transmit)
                {
                    txt1 += "\nsuccesful_transmit\n";
                }
                else
                {
                    txt1 += "\ntransmit_fail\n";
                }

            }
            return txt1;

        }

        public string Protocol_V2d0_disconnect()
        {
            string X = "";
            if (port != null)
            {
                X = ReadALL();
                ClosePort_();
                port.Dispose();
            }
            return X;
        }

        /*
         * 
        https://alexgyver.ru/lessons/crc/

        CRC (cyclic redundancy code) – циклический избыточный код. Алгоритм тоже выдаёт некое “число” при прохождении через него потока байтов, но учитывает все предыдущие данные при расчёте. Как работает данный алгоритм мы рассматривать не будем, об этом можно почитать на Википедии или здесь. Рассмотрим реализацию CRC 8 бит по стандарту Dallas, он используется в датчиках этой фирмы (например DS18b20 и домофонные ключи iButton). Данная реализация должна работать на всех платформах, так как это чисто C++ без привязки к архитектуре (компилятор сам разберётся):
byte crc8(byte *buffer, byte size) {
  byte crc = 0;
  for (byte i = 0; i < size; i++) {
    byte data = buffer[i];
    for (int j = 8; j > 0; j--) {
      crc = ((crc ^ data) & 1) ? (crc >> 1) ^ 0x8C : (crc >> 1);
      data >>= 1;
    }
  }
  return crc;
}

        */

        byte crc8(byte[] buffer, int size)
        {
            byte crc = 0;
            for (byte i = 0; i < size; i++)
            {
                byte data = buffer[i];
                for (int j = 8; j > 0; j--)
                {
                    int tmp = (((int)crc ^ (int)data) & 1);
                    int tmp2 = (tmp != 0) ? ((int)crc >> 1) ^ 0x8C : ((int)crc >> 1);
                    crc = (byte)tmp2;
                    data >>= 1;
                }
            }
            return crc;
        }

    }
}
