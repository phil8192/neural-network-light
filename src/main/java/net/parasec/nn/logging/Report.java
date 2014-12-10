package net.parasec.nn.logging;

import net.parasec.nn.training.TrainingReport;

public final class Report {
  private static final Logger LOG = Logger.getLogger(Report.class);
  private static final String FILE = "errors.csv";

  public static void dump(final TrainingReport tr, final String dir) {
    final String file = dir + "/" + FILE;
    LOG.info("dumping errors to: " + file);
  }
}

