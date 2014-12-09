package net.parasec.nn;

import java.io.FileReader;
import java.io.BufferedReader;
import java.io.IOException;
import java.util.List;
import java.util.LinkedList;

public final class DataLoader {
  private static final Logger LOG = Logger.getLogger(DataLoader.class);
  private static final String DELIMITER = ",";

  /**
   * load training-instance list from a csv. 
   * @param outputLength output vector length. for example,
   * for instance [a,b,c,d,e], outputLength 2 =
   * input vector: [a,b,c] output vector: [d,e].
   */ 
  public List<TrainingInstance> loadCsv(final String file, 
      final int outputLength) {
    try {
      final List<TrainingInstance> instances 
          = new LinkedList<TrainingInstance>();
      final BufferedReader in = new BufferedReader(new FileReader(file));
      try {
        final long l = System.currentTimeMillis();
        String line;
        while((line = in.readLine()) != null) {
          if(line.contains(DELIMITER)) {
            final String[] split = line.split(DELIMITER);
            instances.add(parseInstance(split, outputLength));
          }
        }
        LOG.info("loaded " + instances.size() + " training instances in " +
            (System.currentTimeMillis()-l) + "ms.");
        return instances;    
      } finally {
        in.close();
      }
    } catch(IOException e) {
      LOG.error(e, e);
    }
    return null;
  } 

  /**
   * convert csv row into training-instance
   */ 
  private TrainingInstance parseInstance(final String[] line, 
      final int outputLength) {
    final double[] input = new double[line.length-outputLength];
    final double[] output = new double[outputLength];
    for(int i = 0, len = input.length; i < len; i++)
      input[i] = Double.parseDouble(line[i]);
    final int offset = input.length;
    for(int i = offset, len = line.length; i < len; i++)
      output[i-offset] = Double.parseDouble(line[i]);
    return new TrainingInstance(input, output); 
  }
}

