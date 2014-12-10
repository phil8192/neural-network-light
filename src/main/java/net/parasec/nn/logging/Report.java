package net.parasec.nn.logging;

import net.parasec.nn.training.TrainingReport;

import java.io.OutputStream;
import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.File;
import java.io.IOException;

public final class Report {
  private static final Logger LOG = Logger.getLogger(Report.class);
  private static final String FILE = "errors.csv";


  public static void dump(final TrainingReport tr, final String dir) {
    double[] trainingError = tr.getTrainingError();
    double[] testingError = tr.getTestingError();
    if(trainingError == null && testingError == null) return;
    if(trainingError == null) trainingError = new double[testingError.length];
    if(testingError == null) testingError = new double[trainingError.length];
    final String file = dir + "/" + FILE;
    final long l = System.currentTimeMillis();
    LOG.info("dumping errors to: " + file);
    try {
      OutputStream os = null;
      try {
        os = new BufferedOutputStream(new FileOutputStream(new File(file)));
        for(int i = 0, len = trainingError.length; i < len; i++)
          os.write(getLine(trainingError[i], testingError[i]));  
      } finally {
        if(os != null)
          os.close();
      }
    } catch(final IOException e) {
      LOG.error(e, e);
    }
    LOG.info("wrote data in " + (System.currentTimeMillis()-l) + "ms");
  }

  private static byte[] getLine(final double trainingError, 
      final double testingError) {
    return (String.format("%.5f", trainingError) + "," +
            String.format("%.5f", testingError) + "\n").getBytes();
  }
}

