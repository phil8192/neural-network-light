package net.parasec.nn;

import java.util.Random;

/**
 * utility functions.
 */
public final class MathUtil {
  //
  // note: sigmoid(double) self time = 42.61%.
  // sigmoid calculation is the main hotspot; this is due to the expensive
  // exp function. perhaps the accuracy is not necessary. take a look at
  // http://code.google.com/p/fastapprox/ (approximate and vectorised versions 
  // of functions commonly used in machine learning).
  //
  // this, from: seems good.
  // martin.ankerl.com/2007/02/11/optimized-exponential-functions-for-java/
  //
  //
  //
  private static double fastExp1(final double x) {
    final long tmp = (long) (1512775 * x + 1072632447);
    return Double.longBitsToDouble(tmp << 32);
  }

  //----  standard transfer functions ----//
  public static double sigmoid(final double x) {
    return 1/(1+Math.exp(-x));
  }

  public static double fastSigmoid(final double x) {
    return 1/(1+fastExp1(-x));
  }

  public static double tanh(final double x) {
    return Math.tanh(x);
  }
  //---- ----//

  public static double sigmoidDerivative(final double sOutput) {
    return sOutput*(1-sOutput);
  }

  public static double tanhDerivative(final double thOutput) {
    return 1-(thOutput*thOutput);
  }

  public static double getRandom(final Random prng, final double min, 
      final double max) {
    return prng.nextDouble()*(max-min)+min;
  }

}

