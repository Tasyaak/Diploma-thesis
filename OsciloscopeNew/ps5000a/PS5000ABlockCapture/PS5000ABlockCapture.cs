
using System;
using System.Windows.Forms;

namespace PS5000A
{
    static class PS5000ABlockCapture
    {
        [STAThread]
        static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new PS5000ABlockForm());
        }
    }
}
