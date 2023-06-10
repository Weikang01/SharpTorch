using System;

public static class Util
{
    public static void Break(string note = null)
    {
		try
		{
            int a = 0;
            a /= a;
        }
		catch (System.DivideByZeroException)
        {
            throw new Exception(note);
        }
    }
}
