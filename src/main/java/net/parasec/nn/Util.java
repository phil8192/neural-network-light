package net.parasec.nn;

public final class Util {

  public static String vectorToString(final double[] vector) {
    final int len = (vector != null) ? vector.length - 1 : -1;
    if(len < 0) 
      return "[]";
    final StringBuilder sb = new StringBuilder().append("[ ");
    for(int i = 0; i < len; i++)
      sb.append(String.format("%.4f", vector[i])).append(", ");
    return sb.append(String.format("%.4f", vector[len])).append(" ]")
        .toString();
  }
}

