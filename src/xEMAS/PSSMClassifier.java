/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    PSSMClassifier.java
 *    Copyright (C) 2009 Yasser EL-Manzalawy
 *
 */

/*
 * This programs builds a PSSMClassifier from a set of sequences
 * Modifications:
 * 1- Background sequences can be used to estimate background probabilities.
 * 2- The sigmoid function is used to return scores in [0,1]
 *
 */
package epit.classifiers.matrix;

import weka.classifiers.Classifier;
import weka.core.*;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.Capabilities.Capability;
import java.util.Enumeration;
import java.util.Vector;

/**
 * <!-- globalinfo-start --> Builds a PSSMClassifier from a set of sequences.
 * For more information see:<br>
 *
 * <p>References: <br>
 * Henikoff, J. and Henikoff, S. (1996). Using substitution probabilities to improve position specific scoring matrices. Bioinformatics, 12:135-143.<br>
 * <p/> <!-- globalinfo-end -->
 *
 * <!-- options-start --> Valid options are: <p/>
 *
 * <pre>
 * -N if true, then the negative training data will be used for estimating the
 *    background probabilities. Else, a uniform distribution is assumed.
 * </pre>
 *
 * <!-- options-end -->
 *
 * @author Yasser EL-Manzalawy (yasser@cs.iastate.edu)
 * @version $Revision: 0.9 $
 */
public class PSSMClassifier extends Classifier implements OptionHandler,
        WeightedInstancesHandler {

    /** for serialization */
    static final long serialVersionUID = 5995231201785697655L;
    /** Standard 20 amino acids **/
    private String m_aa = "ACDEFGHIKLMNPQRSTVWY";
    /** BLOSUM62 matrix q_{ia} **/
    // available at:
    // ftp://ftp.ncbi.nih.gov/repository/blocks/unix/blosum/BLOSUM/blosum62.qij
    private double[][] m_q = {
        {0.0215, 0.0016, 0.0022, 0.003, 0.0016, 0.0058, 0.0011, 0.0032,
            0.0033, 0.0044, 0.0013, 0.0019, 0.0022, 0.0019, 0.0023,
            0.0063, 0.0037, 0.0051, 4.00E-04, 0.0013},
        {0.0023, 4.00E-04, 0.0016, 0.0027, 9.00E-04, 0.0017, 0.0012,
            0.0012, 0.0062, 0.0024, 8.00E-04, 0.002, 0.001, 0.0025,
            0.0178, 0.0023, 0.0018, 0.0016, 3.00E-04, 9.00E-04},
        {0.0019, 4.00E-04, 0.0037, 0.0022, 8.00E-04, 0.0029, 0.0014,
            0.001, 0.0024, 0.0014, 5.00E-04, 0.0141, 9.00E-04, 0.0015,
            0.002, 0.0031, 0.0022, 0.0012, 2.00E-04, 7.00E-04},
        {0.0022, 4.00E-04, 0.0213, 0.0049, 8.00E-04, 0.0025, 0.001,
            0.0012, 0.0024, 0.0015, 5.00E-04, 0.0037, 0.0012, 0.0016,
            0.0016, 0.0028, 0.0019, 0.0013, 2.00E-04, 6.00E-04},
        {0.0016, 0.0119, 4.00E-04, 4.00E-04, 5.00E-04, 8.00E-04, 2.00E-04,
            0.0011, 5.00E-04, 0.0016, 4.00E-04, 4.00E-04, 4.00E-04,
            3.00E-04, 4.00E-04, 0.001, 9.00E-04, 0.0014, 1.00E-04,
            3.00E-04},
        {0.0019, 3.00E-04, 0.0016, 0.0035, 5.00E-04, 0.0014, 0.001,
            9.00E-04, 0.0031, 0.0016, 7.00E-04, 0.0015, 8.00E-04,
            0.0073, 0.0025, 0.0019, 0.0014, 0.0012, 2.00E-04, 7.00E-04},
        {0.003, 4.00E-04, 0.0049, 0.0161, 9.00E-04, 0.0019, 0.0014,
            0.0012, 0.0041, 0.002, 7.00E-04, 0.0022, 0.0014, 0.0035,
            0.0027, 0.003, 0.002, 0.0017, 3.00E-04, 9.00E-04},
        {0.0058, 8.00E-04, 0.0025, 0.0019, 0.0012, 0.0378, 0.001, 0.0014,
            0.0025, 0.0021, 7.00E-04, 0.0029, 0.0014, 0.0014, 0.0017,
            0.0038, 0.0022, 0.0018, 4.00E-04, 8.00E-04},
        {0.0011, 2.00E-04, 0.001, 0.0014, 8.00E-04, 0.001, 0.0093,
            6.00E-04, 0.0012, 0.001, 4.00E-04, 0.0014, 5.00E-04, 0.001,
            0.0012, 0.0011, 7.00E-04, 6.00E-04, 2.00E-04, 0.0015},
        {0.0032, 0.0011, 0.0012, 0.0012, 0.003, 0.0014, 6.00E-04, 0.0184,
            0.0016, 0.0114, 0.0025, 0.001, 0.001, 9.00E-04, 0.0012,
            0.0017, 0.0027, 0.012, 4.00E-04, 0.0014},
        {0.0044, 0.0016, 0.0015, 0.002, 0.0054, 0.0021, 0.001, 0.0114,
            0.0025, 0.0371, 0.0049, 0.0014, 0.0014, 0.0016, 0.0024,
            0.0024, 0.0033, 0.0095, 7.00E-04, 0.0022},
        {0.0033, 5.00E-04, 0.0024, 0.0041, 9.00E-04, 0.0025, 0.0012,
            0.0016, 0.0161, 0.0025, 9.00E-04, 0.0024, 0.0016, 0.0031,
            0.0062, 0.0031, 0.0023, 0.0019, 3.00E-04, 0.001},
        {0.0013, 4.00E-04, 5.00E-04, 7.00E-04, 0.0012, 7.00E-04, 4.00E-04,
            0.0025, 9.00E-04, 0.0049, 0.004, 5.00E-04, 4.00E-04,
            7.00E-04, 8.00E-04, 9.00E-04, 0.001, 0.0023, 2.00E-04,
            6.00E-04},
        {0.0016, 5.00E-04, 8.00E-04, 9.00E-04, 0.0183, 0.0012, 8.00E-04,
            0.003, 9.00E-04, 0.0054, 0.0012, 8.00E-04, 5.00E-04,
            5.00E-04, 9.00E-04, 0.0012, 0.0012, 0.0026, 8.00E-04,
            0.0042},
        {0.0022, 4.00E-04, 0.0012, 0.0014, 5.00E-04, 0.0014, 5.00E-04,
            0.001, 0.0016, 0.0014, 4.00E-04, 9.00E-04, 0.0191,
            8.00E-04, 0.001, 0.0017, 0.0014, 0.0012, 1.00E-04, 5.00E-04},
        {0.0063, 0.001, 0.0028, 0.003, 0.0012, 0.0038, 0.0011, 0.0017,
            0.0031, 0.0024, 9.00E-04, 0.0031, 0.0017, 0.0019, 0.0023,
            0.0126, 0.0047, 0.0024, 3.00E-04, 0.001},
        {0.0037, 9.00E-04, 0.0019, 0.002, 0.0012, 0.0022, 7.00E-04,
            0.0027, 0.0023, 0.0033, 0.001, 0.0022, 0.0014, 0.0014,
            0.0018, 0.0047, 0.0125, 0.0036, 3.00E-04, 9.00E-04},
        {4.00E-04, 1.00E-04, 2.00E-04, 3.00E-04, 8.00E-04, 4.00E-04,
            2.00E-04, 4.00E-04, 3.00E-04, 7.00E-04, 2.00E-04, 2.00E-04,
            1.00E-04, 2.00E-04, 3.00E-04, 3.00E-04, 3.00E-04, 4.00E-04,
            0.0065, 9.00E-04},
        {0.0013, 3.00E-04, 6.00E-04, 9.00E-04, 0.0042, 8.00E-04, 0.0015,
            0.0014, 0.001, 0.0022, 6.00E-04, 7.00E-04, 5.00E-04,
            7.00E-04, 9.00E-04, 0.001, 9.00E-04, 0.0015, 9.00E-04,
            0.0102},
        {0.0051, 0.0014, 0.0013, 0.0017, 0.0026, 0.0018, 6.00E-04, 0.012,
            0.0019, 0.0095, 0.0023, 0.0012, 0.0012, 0.0012, 0.0016,
            0.0024, 0.0036, 0.0196, 4.00E-04, 0.0015}};
    /** Q_i **/
    private double[] m_Q;
    /** motif length **/
    private int m_Length;
    /** PSSMClassifier matrix **/
    private double[][] m_PSSM;
    /** If true, the algorithm will use negative data to estimate background probabilities. **/
    protected boolean m_UseNegativeData = true;

    /**
     * Sets the model file parameter tip text in the Weka GUI.
     * @return tip text.
     */
    public String useNegativeDataTipText() {
        return "Whether to use negative data to estimate background probabilities.";
    }

    /**
     * Sets useNegativeData parameter.
     * @param b if true, then negative data will be used to estimate background probabilities.
     *          Otherwise, a uniform background is assumed.
     */
    public void setUseNegativeData(boolean b) {
        m_UseNegativeData = b;
    }

    /**
     * Gets the current setting of useNegativeData.
     * @return The current value of useNegativeData.
     */
    public boolean getUseNegativeData() {
        return m_UseNegativeData;
    }

    /**
     * Returns an enumeration describing the available options.
     *
     * @return an enumeration of all the available options.
     */
    public Enumeration listOptions() {

        Vector newVector = new Vector(1);
        newVector.addElement(new Option(
                "\twhether to use negative data for background frequencies.\n",
                "N", 0, "-N"));

        return newVector.elements();
    }

    /**
     * Gets the current settings of the Classifier.
     *
     * @return an array of strings suitable for passing to setOptions
     */
    public String[] getOptions() {

        String[] options = new String[1];
        int current = 0;

        if (m_UseNegativeData) {
            options[current++] = "-N";
        }

        while (current < options.length) {
            options[current++] = "";
        }

        return options;
    }

    /**
     * Parses a given list of options. Valid options are:<p>
     * 
     * -N <br>
     * if true, then the negative training data will be used for estimating the
     *    background probabilities. Else, a uniform distribution is assumed.<p>
     * 
     * @param options
     * @throws java.lang.Exception
     */
    public void setOptions(String[] options) throws Exception {
        boolean b = Utils.getFlag('N', options);
        setUseNegativeData(b);
    }

    /**
     * Returns a string describing this classifier
     *
     * @return a description of the classifier suitable for displaying in the
     *         explorer/experimenter gui
     */
    public String globalInfo() {
        return "Class for buiding PSSMClassifier matrix By Yasser EL-Manzalawy.\n";
    }

    /**
     * Returns default capabilities of the classifier.
     *
     * @return the capabilities of this classifier
     */
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();

        // class
        result.disableAllClasses();
        result.disableAllClassDependencies();
        result.enable(Capability.BINARY_CLASS);

        // attributes
        result.enable(Capability.STRING_ATTRIBUTES);

        return result;
    }

    /**
     * Generates the classifier.
     *
     * @param instances
     *            set of instances serving as training data
     * @exception Exception
     *                if the classifier has not been generated successfully
     */
    public void buildClassifier(Instances instances) throws Exception {

        // can classifier handle the data?
        getCapabilities().testWithFail(instances);

        // remove instances with missing class
        instances = new Instances(instances);
        instances.deleteWithMissingClass();

        // init Q_i
        m_Q = new double[m_aa.length()];
        for (int i = 0; i < m_aa.length(); i++) {
            m_Q[i] = 0; // init
            for (int a = 0; a < m_aa.length(); a++) {
                m_Q[i] += m_q[i][a];
            }
        }

        // init m_Length
        m_Length = instances.instance(0).stringValue(0).length();

        Instances pos = new Instances(instances, 0);
        Instances neg = new Instances(instances, 0);

        // Question: will the instances be added using their weights?
        // Answer: Yes
        Enumeration e = instances.enumerateInstances();
        while (e.hasMoreElements()) {
            Instance inst = (Instance) e.nextElement();
            if (inst.classValue() == 1) {
                pos.add(inst);
            } else {
                neg.add(inst);
            }
        }
        // System.out.print(pos.instance(0).weight()+ "  "); //testing

        if (m_UseNegativeData == false) {
            neg = new Instances(neg, 0);
        }

        double[][] f_ca = estimateProbabilities(pos); // foreground
        // probabilities
        double[][] b_ca = new double[m_Length][m_aa.length()];
        if (neg.numInstances() == 0) {
            double x = m_aa.length();
            for (int c = 0; c < m_Length; c++) {
                for (int a = 0; a < m_aa.length(); a++) {
                    b_ca[c][a] = 1 / x; // uniform background prob
                }
            }
        } else {
            b_ca = estimateProbabilities(neg); // background probabilities
        }

        // compute pssm
        m_PSSM = new double[m_Length][m_aa.length()];
        for (int c = 0; c < m_Length; c++) {
            for (int a = 0; a < m_aa.length(); a++) {
                m_PSSM[c][a] = Math.log(f_ca[c][a] / b_ca[c][a]); // Note: for aa at the center of the window the two freq are the same so the result is zero
            }
        }
        // clear memory for pos and neg
        pos = new Instances(pos, 0);
        neg = new Instances(neg, 0);

    }

    private double[][] estimateProbabilities(Instances instances)
            throws Exception {
        double[][] n_ca = new double[m_Length][m_aa.length()];
        double[][] b_ca = new double[m_Length][m_aa.length()];
        double Nc = 0;
        Enumeration e = instances.enumerateInstances();
        while (e.hasMoreElements()) { // for each sequence
            Instance inst = (Instance) e.nextElement();
            Nc += inst.weight();
            String s = inst.stringValue(0);
            // for each aa
            for (int c = 0; c < m_Length; c++) { // for each column
                int a = m_aa.indexOf(s.substring(c, c + 1));
                if (a < 0) {
                    continue; // skip this non standard aa
                }
                n_ca[c][a] += inst.weight(); // weighted count n_{ca}
            }

        }
        double Bc = Math.sqrt(Nc);

        // compute b_{ca}
        for (int c = 0; c < m_Length; c++) {
            for (int a = 0; a < m_aa.length(); a++) {
                double x = 0.0;
                for (int i = 0; i < m_aa.length(); i++) {
                    x += (n_ca[c][i] * m_q[i][a]) / (Nc * m_Q[i]);
                }
                b_ca[c][a] = Bc * x;
            }
        }

        // compute p_{ca}
        double w1 = Nc / (Nc + Bc);
        double w2 = Bc / (Nc + Bc);
        for (int c = 0; c < m_Length; c++) {
            for (int a = 0; a < m_aa.length(); a++) {
                b_ca[c][a] = (w1 * (n_ca[c][a] / Nc)) + (w2 * (b_ca[c][a]) / Bc);
            }
        }

        return b_ca;
    }

    private double score(Instance instance) throws Exception {
        String s = instance.stringValue(0);
        double score = 0.0;
        for (int c = 0; c < m_Length; c++) {
            int a = m_aa.indexOf(s.substring(c, c + 1));
            if (a < 0) {
                continue;
            }
            score += m_PSSM[c][a];
        }
        return score;
    }

    /**
     * Calculates the class membership probabilities for the given test
     * instance.
     *
     * @param instance
     *            the instance to be classified
     * @return predicted class probability distribution
     * @exception Exception
     *                if there is a problem generating the prediction
     */
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] probs = new double[2];
        String s = instance.stringValue(0);
        double score = 0.0;
        for (int c = 0; c < m_Length; c++) {
            int a = m_aa.indexOf(s.substring(c, c + 1));
            if (a < 0) {
                continue;
            }
            score += m_PSSM[c][a];
        }
        // System.out.println(score);

        probs[1] = 1 / (1 + Math.exp(-1 * score));
        probs[0] = 1 - probs[1];
        // Utils.normalize(probs);
        return probs;
    }

    /*
     * public double [] distributionForInstance(Instance instance) throws
     * Exception { Instance inst = new Instance(2); inst.setDataset(data);
     * inst.setValue(0, score(instance)); m_normalize.input(inst); return
     * m_logistic.distributionForInstance(m_normalize.output()); }
     */
    /**
     * Returns a description of the classifier.
     *
     * @return a description of the classifier as a string.
     */
    public String toString() {

        StringBuffer text = new StringBuffer();

        text.append("PSSMClassifier Classifier by Yasser EL-Manzalawy\n");
        for (int c = 0; c < m_Length; c++) {
            String line = new String();
            for (int a = 0; a < m_aa.length(); a++) {
                line += m_PSSM[c][a] + " ";
            }
            text.append(line + "\n");
        }
        return text.toString();
    }

    public double[] getMatrix() {
        int position = 0;
        int columns = m_aa.length();
        double[] mat = new double[m_Length * columns];
        for (int row = 0; row < m_Length; row++) {
            for (int col = 0; col < columns; col++) {
                mat[position++] = m_PSSM[row][col];
            }
        }
        return mat;
    }

    /**
     * Returns the revision string.
     *
     * @return the revision
     */
    public String getRevision() {
        return RevisionUtils.extract("$Revision: 0.9 $");
    }

    /**
     * Main method for testing this class.
     *
     * @param argv
     *            The options.
     */
    public static void main(String[] argv) {
        runClassifier(new PSSMClassifier(), argv);
    }
}
