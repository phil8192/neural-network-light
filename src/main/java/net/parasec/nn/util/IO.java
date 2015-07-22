package net.parasec.nn.util;

import net.parasec.nn.logging.Logger;

import java.io.ObjectOutputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.FileInputStream;
import java.io.IOException;

/**
 * save and load weights. 
 * just serialise weight matrix.
 */
public final class IO {

  private static final Logger LOG = Logger.getLogger(IO.class);

  public static void dumpWeights(final double[][][] weights, 
      final String filename) {
/*
for(int i = 0; i < weights.length; i++)
  for(int j = 0; j < weights[i].length; j++)
    for(int k = 0; k < weights[i][j].length; k++)
      System.out.println(i+"|"+j+"|"+k+" "+weights[i][j][k]);
*/
    try {
      ObjectOutputStream out = null;
      try {
        LOG.info("saving weights to " + filename);
        out = new ObjectOutputStream(new FileOutputStream(filename));
        out.writeObject(weights); 
      } finally {
        if(out != null)
          out.close();
      }
    } catch(IOException e) {
      LOG.error(e, e);
    } 
  }

  public static double[][][] loadWeights(final String filename) {
    try {
      ObjectInputStream in = null;
      try {
        LOG.info("loading weights from " + filename); 
        in = new ObjectInputStream(new FileInputStream(filename));
        return (double[][][]) in.readObject();
      } finally {
        if(in != null)
          in.close();
      }
    } catch(IOException e) {
      LOG.error(e, e);
    } catch(ClassNotFoundException e) {
      LOG.error(e, e);
    }
    return null;
  }
}

