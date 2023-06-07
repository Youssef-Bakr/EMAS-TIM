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
 *    BalancedClassifier.java
 *    Copyright (C) 2009 Yasser EL-Manzalawy
 *
 */
package epit.classifiers.meta;

import weka.classifiers.RandomizableSingleClassifierEnhancer;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;
import weka.core.RevisionUtils;
import weka.core.WeightedInstancesHandler;
import weka.core.Capabilities.Capability;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

/**
<!-- globalinfo-start --> A meta classifier for training a base classifier on unbalanced data set.
 * Specifically, the training instances in the majority class will be randomized and
 * only a subset of these instances (determined by the parameter R) will be used for training the classifier.
 * Let m be the number of training instances in the minority class, n be the number of training instances in the majority
 * class. This classifier will be trained using the instances in the minority instances plus R*m instances
 * randomly chosen from the training instances in the majority class.
 *
<!-- globalinfo-end -->
 *
 *
<!-- options-start -->
 * Valid options are: <p/>
 *
 *
 * * <pre>
 * -S &lt;int&gt;
 *  Random seed for randomizing the training data.
 * </pre>
 *
 * <pre> -R &lt;num&gt;
 *   Determines the number of training instances in the majority class that will
 *   be used for training the classifier (default = 1).
 * </pre>
 *
 * <pre> -W
 *    The Classifier to be trained/evaluated.
 * </pre>
 *
<!-- options-end -->
 *
 * @author Yasser EL-Manzalawy (yasser@cs.iastate.edu)
 * @version $Revision: 0.9 $
 */
public class BalancedClassifier
        extends RandomizableSingleClassifierEnhancer
        implements WeightedInstancesHandler {

    /** for serialization */
    static final long serialVersionUID = -7378107808933117974L;
    /** Determines the number of training instances in the majority class that will
    be used for training the classifier (default = 1). */
    protected double m_Ratio = 1.0;

    /**
     * Returns a string describing classifier
     * @return a description suitable for
     * displaying in the explorer/experimenter gui
     */
    public String globalInfo() {
        return "Balanced Classifier By Yasser EL-Manzalawy.\n";
    }

    /**
     * Returns an enumeration describing the available options.
     *
     * @return an enumeration of all the available options.
     */
    public Enumeration listOptions() {

        Vector newVector = new Vector();

        newVector.addElement(new Option(
                "\tRatio between majority and minority instances.\n" + "\t(default 1.0)",
                "R", 1, "-R <num>"));

        Enumeration enu = super.listOptions();
        while (enu.hasMoreElements()) {
            newVector.addElement(enu.nextElement());
        }

        return newVector.elements();
    }

    /**
     * Parses a given list of options.
    <!-- options-start -->
     * Valid options are: <p/>
     *
     *
     * * <pre>
     * -S &lt;int&gt;
     *  Random seed for randomizing the training data.
     * </pre>
     *
     * <pre> -R &lt;num&gt;
     *   Determines the number of training instances in the majority class that will
     *   be used for training the classifier (default = 1).
     * </pre>
     *
     * <pre> -W
     *    The Classifier to be trained/evaluated.
     * </pre>
     *
    <!-- options-end -->
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    public void setOptions(String[] options) throws Exception {

        String thresholdString = Utils.getOption('R', options);
        if (thresholdString.length() != 0) {
            setRatio(Double.parseDouble(thresholdString));
        } else {
            setRatio(1.0);	// default
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
        String[] options = new String[superOptions.length + 2];

        int current = 0;

        options[current++] = "-R";
        options[current++] = "" + getRatio();


        System.arraycopy(superOptions, 0, options, current,
                superOptions.length);
        return options;
    }

    /**
     * Returns the tip text for this property
     * @return tip text for this property suitable for
     * displaying in the explorer/experimenter gui
     */
    public String ratioTipText() {
        return "Ratio between majority and minority instances in the training data passed to the learner.";
    }

    /**
     * Sets the Ratio parameter. If m is the number of training instances in the minority class and n is the number of training instances in the majority
     * class. This classifier will be trained using the instances in the minority instances plus R*m instances
     * randomly chosen from the training instances in the majority class.
     *
     * @param ratio The ratio between the training instances in the majority class to those in the minority class.
     */
    public void setRatio(double ratio) {
        m_Ratio = ratio;
    }

    /**
     * Get the degree of weight thresholding
     *
     * @return the percentage of weight mass used for training
     */
    public double getRatio() {
        return m_Ratio;
    }

    /**
     * Returns default capabilities of the classifier.
     *
     * @return      the capabilities of this classifier
     */
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();

        // class
        result.disableAllClasses();
        result.disableAllClassDependencies();
        if (super.getCapabilities().handles(Capability.NOMINAL_CLASS)) {
            result.enable(Capability.NOMINAL_CLASS);
        }
        if (super.getCapabilities().handles(Capability.BINARY_CLASS)) {
            result.enable(Capability.BINARY_CLASS);
        }

        return result;
    }

    /**
     * Balances the training data and builds the classifier on the balanced data.
     *
     * @param data the training data to be used for generating the
     * boosted classifier.
     * @throws Exception if the classifier could not be built successfully
     */
    public void buildClassifier(Instances data) throws Exception {
        Instances data0 = new Instances(data, 0);
        Instances data1 = new Instances(data, 0);

        for (int i = 0; i < data.numInstances(); i++) {
            if (data.instance(i).classValue() == 0) {
                data0.add(data.instance(i));
            } else {
                data1.add(data.instance(i));
            }
        }

        double x0 = data0.numInstances();
        double x1 = data1.numInstances();

        double len = 0;
        if (x0 > x1) {	//class0 is the majority
            len = x1 * m_Ratio;
            if (len > x0) {
                len = x0;
            }
            data0.randomize(new Random(m_Seed));
            for (int i = 0; i < len; i++) {
                data1.add(data0.instance(i));
            }
            m_Classifier.buildClassifier(data1);
        } else {			//class1 is the majority
            len = x0 * m_Ratio;
            if (len > x1) {
                len = x1;
            }
            data0.randomize(new Random(m_Seed));
            for (int i = 0; i < len; i++) {
                data0.add(data1.instance(i));
            }
            m_Classifier.buildClassifier(data0);
        }
        // free memory
        data0 = new Instances(data, 0);
        data1 = new Instances(data, 0);
    }

    /**
     * Calculates the class membership probabilities for the given test instance.
     *
     * @param instance the instance to be classified
     * @return predicted class probability distribution
     * @throws Exception if instance could not be classified
     * successfully
     */
    public double[] distributionForInstance(Instance instance)
            throws Exception {

        return m_Classifier.distributionForInstance(instance);

    }

    /**
     * Returns description of the boosted classifier.
     *
     * @return description of the boosted classifier as a string
     */
    public String toString() {
        String text = "Balanced Classifier by Yasser EL-Manzalawy \n" + "Ratio " + m_Ratio + "\n" + m_Classifier.toString();

        return text;
    }

    /**
     * Returns the revision string.
     *
     * @return		the revision
     */
    public String getRevision() {
        return RevisionUtils.extract("$Revision: 0.9 $");
    }

    /**
     * Main method for testing this class.
     *
     * @param argv The options.
     */
    public static void main(String[] argv) {
        runClassifier(new BalancedClassifier(), argv);
    }
}

