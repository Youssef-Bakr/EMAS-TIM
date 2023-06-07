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
 *    MILESReg.java
 *    Copyright (C) 2009 Yasser EL-Manzalawy
 *
 */
package epit.classifiers.mi;

import weka.core.*;
import weka.classifiers.SingleClassifierEnhancer;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.Capabilities.Capability;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

/**
 * <!-- globalinfo-start -->
 * MILESReg is an adaptation of MILES multiple-instance learning algorithm for
 * multiple-instance regression on bags of 9-mers.<br>
 *
 * <p>References: <br>
 * Chen, Y., Bi, J., and Wang, J. (2006). MILES: multiple-instance learning via embedded instance selection. IEEE Trans. Pattern Anal. Mach. Intell., 28:1931-1947.<br>
 * <p/> <!-- globalinfo-end  -->
 *
 * <!-- options-start --> Valid options are: <p/>
 * <pre>
 * -W
 *  Regression algorithm.
 * </pre>
 *
 * <pre>
 * -P &lt;num&gt;
 * Percent of the set of all 9-mers that will be used for mapping each bag of 9-mers
 * into a meta-instance. (default = -1 which implies that 100% of the all 9-mers
 * will be used for the mapping).
 * </pre>
 *
 * <pre>
 * -S &lt;int&gt;
 *  Random seed.
 * </pre>
 * <!-- options-end -->
 *
 * @author Yasser EL-Manzalawy (yasser@cs.iastate.edu)
 * @version $Revision: 0.9 $
 */
public class MILESReg extends SingleClassifierEnhancer implements
        OptionHandler, WeightedInstancesHandler {

    /** for serialization */
    static final long serialVersionUID = -2368937577670527151L;
    /** set of all 9-mer windows in the training data **/
    private Vector m_Windows;
    /** BLOSUM62 matrix **/
    private double[][] sub_mat = { // http://expasy.org/cgi-bin/blosum.pl
        {4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0,
            -3, -2, 0},
        {-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3,
            -2, -3},
        {-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2,
            -3},
        {-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1,
            -4, -3, -3},
        {0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1,
            -2, -2, -1},
        {-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2,
            -1, -2},
        {-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3,
            -2, -2},
        {0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2,
            -2, -3, -3},
        {-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2,
            -2, 2, -3},
        {-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1,
            -3, -1, 3},
        {-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1,
            -2, -1, 1},
        {-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3,
            -2, -2},
        {-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1,
            -1, -1, 1},
        {-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2,
            1, 3, -1},
        {-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1,
            -1, -4, -3, -2},
        {1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3,
            -2, -2},
        {0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5,
            -2, -2, 0},
        {-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3,
            -2, 11, 2, -3},
        {-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2,
            2, 7, -1},
        {0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0,
            -3, -1, 4}};
    /** amino acids as ordered in BLOSUM62 **/
    private String aa = "ARNDCQEGHILKMFPSTWYV";
    /** num of attributes = size of m_Windows **/
    private int m_NumAttributes;
    /** data in the embedded instance space **/
    private Instances m_Instances;
    /**
     * The size of the features as a percent of the 9-mer windows in the
     * training data
     */
    /** Percent of the set of all 9-mers that will be used for mapping each bag of 9-mers
    into a meta-instance. If the seed parameter (-S) is -1, then 100% of the all 9-mers
    will be used for the mapping. */
    protected int m_DataSizePercent = 10;
    /** Random seed for extracting a random subset of the set of all all 9-mers.
     * Default = -1 which implies that all all 9-mers will be used. */
    protected int m_Seed = -1;

    /**
     * Returns a string describing this attribute evaluator
     *
     * @return a description of the evaluator suitable for displaying in the
     *         explorer/experimenter gui
     */
    public String globalInfo() {
        return " An adaptation of MILES for Multiple-Instance regression over bags of amino acid sequences. " + "For more information see:\n" + "Chen, Y., Bi, J., and Wang, J. (2006). MILES: multiple-instance learning via embedded" + " instance selection. IEEE Trans. Pattern Anal. Mach. Intell., 28:1931{1947.";
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     *         explorer/experimenter gui
     */
    public String dataSizePercentTipText() {
        return "size of the features as a percent of the 9-mer windows in the training datat.";
    }

    /**
     * Gets the size of each bag, as a percentage of the training set size.
     *
     * @return the bag size, as a percentage.
     */
    public int getDataSizePercent() {
        return m_DataSizePercent;
    }

    /**
     * Sets the size of each bag, as a percentage of the training set size.
     *
     * @param newDataSizePercent
     *            the bag size, as a percentage.
     */
    public void setDataSizePercent(int newDataSizePercent) {

        m_DataSizePercent = newDataSizePercent;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     *         explorer/experimenter gui
     */
    public String seedTipText() {
        return "Seed to randomize the training 9-mers.";
    }

    /**
     * Gets the size of each bag, as a percentage of the training set size.
     *
     * @return the bag size, as a percentage.
     */
    public int getSeed() {

        return m_Seed;
    }

    /**
     * Sets the size of each bag, as a percentage of the training set size.
     *
     * @param newDataSizePercent
     *            the bag size, as a percentage.
     */
    public void setSeed(int seed) {

        m_Seed = seed;
    }

    /**
     * Returns an enumeration describing the available options.
     *
     * <!-- options-start --> Valid options are: <p/>
     * <pre>
     * -W
     *  Regression algorithm.
     * </pre>
     *
     * <pre>
     * -P &lt;num&gt;
     * Percent of the set of all 9-mers that will be used for mapping each bag of 9-mers
     * into a meta-instance. (default = -1 which implies that 100% of the all 9-mers
     * will be used for the mapping).
     * </pre>
     *
     * <pre>
     * -S &lt;int&gt;
     *  Random seed.
     * </pre>
     * <!-- options-end -->
     *
     * 
     * @return an enumeration of all the available options.
     */
    public Enumeration listOptions() {

        Vector newVector = new Vector(4);

        newVector.addElement(new Option(
                "\tSeed. " + "(default = -1, ie. use all the instances for feature mapping)\n",
                "S", 1, "-S <num>"));

        newVector.addElement(new Option(
                "\tSize of features, as a percentage of the\n" + "\t9-mers in training set size. (default 10)", "P",
                1, "-P <num>"));

        Enumeration enu = super.listOptions();
        while (enu.hasMoreElements()) {
            newVector.addElement(enu.nextElement());
        }
        return newVector.elements();
    }

    /**
     * Parses a given list of options. <p/>
     *
 * <!-- options-start --> Valid options are: <p/>
 * <pre>
 * -W
 *  Regression algorithm.
 * </pre>
 *
 * <pre>
 * -P &lt;num&gt;
 * Percent of the set of all 9-mers that will be used for mapping each bag of 9-mers
 * into a meta-instance. (default = -1 which implies that 100% of the all 9-mers
 * will be used for the mapping).
 * </pre>
 *
 * <pre>
 * -S &lt;int&gt;
 *  Random seed.
 * </pre>
 * <!-- options-end -->
     *
     *
     * @param options
     *            the list of options as an array of strings
     * @throws Exception
     *             if an option is not supported
     */
    public void setOptions(String[] options) throws Exception {

        String seedStr = Utils.getOption('S', options);
        if (seedStr.length() != 0) {
            setSeed(Integer.parseInt(seedStr));
        } else {
            setSeed(-1);
        }

        String dataSize = Utils.getOption('P', options);
        if (dataSize.length() != 0) {
            setDataSizePercent(Integer.parseInt(dataSize));
        } else {
            setDataSizePercent(10);
        }

        super.setOptions(options);
    }

    /**
     * Gets the current settings of the Classifier.
     *
     * @return an array of strings suitable for passing to setOptions
     */
    public String[] getOptions() {
        String[] superOptions = super.getOptions();
        String[] options = new String[superOptions.length + 4];

        int current = 0;
        options[current++] = "-S";
        options[current++] = "" + getSeed();

        options[current++] = "-P";
        options[current++] = "" + getDataSizePercent();

        System.arraycopy(superOptions, 0, options, current,
                superOptions.length);

        current += superOptions.length;
        while (current < options.length) {
            options[current++] = "";
        }
        return options;
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
        result.enable(Capability.NUMERIC_CLASS);
        result.enable(Capability.DATE_CLASS);

        // attributes
        result.enable(Capability.STRING_ATTRIBUTES);

        return result;
    }

    /**
     * Build the classifier on the supplied data
     *
     * @param data
     *            the training data
     * @throws Exception
     *             if the classifier could not be built successfully
     */
    public void buildClassifier(Instances data) throws Exception {

        // can classifier handle the data?
        getCapabilities().testWithFail(data);

        // remove instances with missing class
        Instances newData = new Instances(data);
        newData.deleteWithMissingClass();

        // get set of all 9-mer windows
        data2Windows(data);
        if (m_Seed != -1) { // use all 9-mers
            FastVector atts = new FastVector(1);
            atts.addElement(new Attribute("9mers", (FastVector) null));
            Instances tmpInstances = new Instances("tmp", atts, 0);
            for (int i = 0; i < m_Windows.size(); i++) {
                Instance inst = new Instance(1);
                inst.setDataset(tmpInstances);
                inst.setValue(0, (String) m_Windows.elementAt(i));
                tmpInstances.add(inst);
            }
            m_Windows = new Vector();
            tmpInstances.randomize(new Random(m_Seed));
            double len = (tmpInstances.numInstances() * m_DataSizePercent) / 100;
            for (int i = 0; i < len; i++) {
                m_Windows.add(tmpInstances.instance(i).stringValue(0));
            }
            tmpInstances = null;
        }

        m_NumAttributes = m_Windows.size();

        if (m_Debug) {
            System.out.println("Total No. of features: " + m_NumAttributes);
        }

        // create new Instances
        FastVector attInfo = new FastVector(m_NumAttributes + 1);
        for (int i = 0; i < m_NumAttributes; i++) {
            attInfo.addElement(new Attribute("a" + i));
        }
        Attribute labelAttr = (Attribute) data.instance(0).classAttribute().copy();
        attInfo.addElement(labelAttr);
        m_Instances = new Instances("propData", attInfo, 0);
        m_Instances.setClassIndex(m_Instances.numAttributes() - 1);

        Enumeration enumr = data.enumerateInstances();
        while (enumr.hasMoreElements()) {
            Instance temp = (Instance) enumr.nextElement();
            m_Instances.add(map(temp));
        }

        // build base classifier
        m_Classifier.buildClassifier(m_Instances);
    }

    /**
     * Maps a bag into a single meta-instance.
     * @param instance
     * @return
     * @throws java.lang.Exception
     */
    private Instance map(Instance instance) throws Exception {
        Instance tmp = new Instance(m_NumAttributes + 1);
        tmp.setDataset(m_Instances);
        String seq = instance.stringValue(0);
        for (int i = 0; i < m_NumAttributes; i++) {
            tmp.setValue(i, distance((String) m_Windows.elementAt(i), seq));
        }
        tmp.setClassValue(instance.classValue());
        return tmp;
    }

    /**
     * Classify an instance.
     *
     * @param inst
     *            the instance to predict
     * @return a prediction for the instance
     * @throws Exception
     *             if an error occurs
     */
    public double classifyInstance(Instance inst) throws Exception {
        Instance tmp = map(inst);
        return m_Classifier.classifyInstance(tmp);
    }

    /**
     * Returns textual description of the classifier.
     *
     * @return a description of the classifier as a string
     */
    public String toString() {
        return "MILESReg By Yasser EL-Manzalawy.\n";
    }

    /****
     * Applies a sliding window of length to seq.
     *
     * @param seq
     * @param length
     * @return vector of length-mer windows
     */
    private Vector sequence2Windows(String seq, int length) {
        int len = seq.length();
        Vector w = new Vector();
        for (int start = 0; start <= (len - length); start++) {
            w.add(seq.substring(start, start + length));
        }
        return w;
    }

   /**
    * Extracts the set of all 9-mers.
    * @param data data set of peptides.
    */
    private void data2Windows(Instances data) {
        m_Windows = new Vector();
        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance tmp = (Instance) instEnum.nextElement();
            m_Windows.addAll(sequence2Windows(tmp.stringValue(0), 9));
        }
    }

    /**
     * Computes the distance between two windows using BLOSUM62 matrix
     *
     * @param w1
     * @param w2
     * @return
     */
    private double distanceBLOSUM62(String w1, String w2) {
        double d = 0;
        int ind1, ind2;
        for (int i = 0; i < w1.length(); i++) {
            ind1 = aa.indexOf(w1.substring(i, i + 1));
            ind2 = aa.indexOf(w2.substring(i, i + 1));
            // System.out.println("index 1 " + ind1 + " " + ind2);
            if (ind1 == -1 || ind2 == -1); else {
                d += sub_mat[ind1][ind2];
            }
        }
        if (d <= 0) {
            return 1;
        } else {
            return (1 / d);
        }
    }

    private double distance(String w, String seq) {
        Vector ws = sequence2Windows(seq, 9);
        double min = Double.MAX_VALUE;
        for (int i = 0; i < ws.size(); i++) {
            double d = distanceBLOSUM62(w, (String) ws.elementAt(i));
            if (d < min) {
                min = d;
            }
        }
        return min;
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
     *           The options.
     */
    public static void main(String[] argv) {
        runClassifier(new MILESReg(), argv);
    }
}
