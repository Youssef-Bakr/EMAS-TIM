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
 *    SequenceComposition.java
 *    Copyright (C) 2009 Yasser EL-Manzalawy
 *
 */
package epit.filters.unsupervised.attribute;

import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.filters.Filter;
import weka.filters.UnsupervisedFilter;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.*;

/**
 * <!-- globalinfo-start --> Produces the sequence composition features of input sequences
 * (e.g., the amino acid composition features of protein sequences).
 *  <p/> <!-- globalinfo-end -->
 *
 * <!-- options-start --> Valid options are: <p/>
 *
 * <pre>
 * -A &lt;string&gt;
 *  The alphabet (default standard 20 amino acids)
 * </pre>
 *
 * <pre>
 * -S &lt;num&gt;
 *  The scaling method (default 0)
 *  	0 no weights
 *  	1 beta turns
 *  	2 accessability
 *  	3 flexibility
 *  	4 antigenicity
 *  	5 hydrophobicity
 * </pre>
 *
 * <!-- options-end -->
 *
 * @author Yasser EL-Manzalawy (yasser@cs.iastate.edu)
 * @version $Revision: 0.9 $
 */
public class SequenceComposition extends Filter implements UnsupervisedFilter,
        OptionHandler {

    /** for serialization */
    static final long serialVersionUID = 3119607037607101160L;  //TODO:
    /** The sequence alphabet (default = standard 20 amino acids). */
    protected String m_AA = "ACDEFGHIKLMNPQRSTVWY";
    /** The weights for scaling attributes based on a specified AA index. */
    private double[] m_Weights;
    /** The selected scaling method */
    protected int m_PropensityScale = SCALE_DEFAULT;
    /** no scaling required */
    public static final int SCALE_DEFAULT = 0;
    /** scale the frequency of each amino acid using the beta-turn index of that amino acid. */
    public static final int SCALE_BETA = 1;
    /** scale the frequency of each amino acid using the accessibility index of that amino acid. */
    public static final int SCALE_ACCESSABILITY = 2;
    /** scale the frequency of each amino acid using the flexibility index of that amino acid. */
    public static final int SCALE_FLEXIBILITY = 3;
    /** scale the frequency of each amino acid using the antigenicity index of that amino acid. */
    public static final int SCALE_ANTIGENICITY = 4;
    /** scale the frequency of each amino acid using the hydrophobicity index of that amino acid. */
    public static final int SCALE_HYDRO = 5;
    public static final Tag[] TAGS_WEIGHTMETHOD = {
        new Tag(SCALE_DEFAULT, "no weights"),
        new Tag(SCALE_BETA, "beta turns"),
        new Tag(SCALE_ACCESSABILITY, "accessibility"),
        new Tag(SCALE_FLEXIBILITY, "flexibility"),
        new Tag(SCALE_ANTIGENICITY, "antigenicity"),
        new Tag(SCALE_HYDRO, "hydrophobicity"),};

    /**
     * Default constructor.
     */
    public SequenceComposition() {
        m_Weights = new double[m_AA.length()];
        for (int i = 0; i < m_Weights.length; i++) // set default scale
        {
            m_Weights[i] = 1.0;
        }
    }

    /**
     * Sets the alphabet parameter tip text in the Weka GUI.
     * @return tip text.
     */
    public String alphabetTipText() {
        return "The squence alphabet.";
    }

    /**
     * Gets the sequence alphabet.
     * @return The sequence alphabet.
     */
    public String getAlphabet() {
        return m_AA;
    }

    /**
     * Sets the sequence alphabet.
     * @param alphabet The sequence alphabet.
     */
    public void setAlphabet(String alphabet) {
        m_AA = alphabet;
    }

    /**
     * Sets the Scale parameter tip text in the Weka GUI.
     * @return
     */
    public String propensityScaleTipText() {
        return "The scaling methods.";
    }

    /**
     * Gets the scaling method.
     * @return an integer value specifying the scaling method.
     *      0 no weights
     *  	1 beta turns
     *  	2 accessibility
     *  	3 flexibility
     *  	4 antigenicity
     *  	5 hydrophobicity
     *
     */
    public SelectedTag getPropensityScale() {
        return new SelectedTag(m_PropensityScale, TAGS_WEIGHTMETHOD);
    }

    /**
     * Sets the scaling method.
     *
     * @param newMethod
     *            the new scaling method.
     */
    public void setPropensityScale(SelectedTag newMethod) {
        if (newMethod.getTags() == TAGS_WEIGHTMETHOD) {
            m_PropensityScale = newMethod.getSelectedTag().getID();
        }
    }

    /**
     * Lists valid options for that filter.
     * <!-- options-start --> Valid options are: <p/>
     *
     * <pre>
     * -A &lt;string&gt;
     *  The alphabet (default standard 20 amino acids)
     * </pre>
     *
     * <pre>
     * -S &lt;num&gt;
     *  The scaling method (default 0)
     *  	0 no weights
     *  	1 beta turns
     *  	2 accessability
     *  	3 flexibility
     *  	4 antigenicity
     *  	5 hydrophobicity
     * </pre>
     *
     * <!-- options-end -->
     *
     */
    public Enumeration listOptions() {
        Vector newVector = new Vector(2);
        newVector.addElement(new Option(
                "\tThe scaling methods:\n" + "\t 0. no weights\n" + "\t 1. beta turns\n" + "\t 2. accessability\n" + "\t 3. flexibility\n" + "\t 4. antigenicity\n" + "\t 5. hydrophobicity\n",
                "S", 1, "-S [0|1|2|3|4|5]"));
        newVector.addElement(new Option("\tSets sequence alphabet\n", "A", 1,
                "-A <string>"));
        return newVector.elements();
    }

    /**
     * Sets filter options.
     * -A sequence alphabet (default 20 amino acids).
     * -S scaling method  (default none).
     */
    public void setOptions(String[] options) throws Exception {
        String tmpStr = Utils.getOption('S', options);
        if (tmpStr.length() != 0) {
            setPropensityScale(new SelectedTag(Integer.parseInt(tmpStr),
                    TAGS_WEIGHTMETHOD));
        } else {
            setPropensityScale(new SelectedTag(SCALE_DEFAULT, TAGS_WEIGHTMETHOD));
        }

        tmpStr = Utils.getOption('A', options);
        if (tmpStr.length() != 0) {
            setAlphabet(tmpStr);
            m_Weights = new double[m_AA.length()];
            for (int i = 0; i < m_Weights.length; i++) {
                m_Weights[i] = 1.0;
            }
        } else {
            setAlphabet("ACDEFGHIKLMNPQRSTVWYX");
        }

    }

    /**
     * Returns current values for the filter options.
     */
    public String[] getOptions() {
        String[] options = new String[4];
        int current = 0;
        options[current++] = "-S";
        options[current++] = "" + getPropensityScale();
        options[current++] = "-A";
        options[current++] = "" + getAlphabet();
        return options;
    }

    /**
     * Sets the attribute weights based on the selected scaling method.
     */
    // TODO: add a parameter to allow the user to use any arbitrary scale
    private void loadWeights() {
        if (m_PropensityScale == SCALE_DEFAULT) {
            for (int i = 0; i < m_Weights.length; i++) {
                m_Weights[i] = 1.0;
            }
            return;
        } else if (m_PropensityScale == SCALE_BETA) {
            double[] beta = {0.66, 1.19, 1.46, 0.74, 0.6, 1.56, 0.95, 0.47,
                1.01, 0.59, 0.6, 1.56, 1.52, 0.98, 0.95, 1.43, 0.96, 0.5,
                0.96, 1.14};
            for (int i = 0; i < m_Weights.length; i++) {
                m_Weights[i] = beta[i];
            }

        } else if (m_PropensityScale == SCALE_ACCESSABILITY) {
            double[] access = {0.49, 0.26, 0.81, 0.84, 0.42, 0.48, 0.66, 0.34,
                0.97, 0.4, 0.48, 0.78, 0.75, 0.84, 0.95, 0.65, 0.7, 0.36,
                0.51, 0.76};
            for (int i = 0; i < m_Weights.length; i++) {
                m_Weights[i] = access[i];
            }

        } else if (m_PropensityScale == SCALE_FLEXIBILITY) {
            double[] flex = {1.041, 0.96, 1.033, 1.094, 0.93, 1.142, 0.982,
                1.002, 1.093, 0.967, 0.947, 1.117, 1.055, 1.165, 1.038,
                1.169, 1.073, 0.982, 0.925, 0.961};
            for (int i = 0; i < m_Weights.length; i++) {
                m_Weights[i] = flex[i];
            }

        } else if (m_PropensityScale == SCALE_ANTIGENICITY) {
            double[] antigen = {1.064, 1.412, 0.866, 0.851, 1.091, 0.874,
                1.105, 1.152, 0.93, 1.25, 0.826, 0.776, 1.064, 1.015,
                0.873, 1.012, 0.909, 1.383, 0.893, 1.161};
            for (int i = 0; i < m_Weights.length; i++) {
                m_Weights[i] = antigen[i];
            }

        } else {
            double[] parker = {2.1, 1.4, 10, 7.8, -9.2, 5.7, 2.1, -8, 5.7,
                -9.2, -4.2, 7, 2.1, 6, 4.2, 6.5, 5.2, -3.7, -10, -1.9};

            for (int i = 0; i < m_Weights.length; i++) {
                m_Weights[i] = parker[i];
            }
        }

        // normalize weights to interval [0,1]
        double min = m_Weights[Utils.minIndex(m_Weights)];
        double max = m_Weights[Utils.maxIndex(m_Weights)];
        for (int i = 0; i < m_Weights.length; i++) {
            m_Weights[i] = (m_Weights[i] - min) / (max - min);
        }
    }

    /**
     * Returns a string describing this classifier
     *
     * @return a description of the classifier suitable for displaying in the
     *         explorer/experimenter gui
     */
    public String globalInfo() {
        return "SequenceComposition  filter by Yasser EL-Manzalawy.\n";
    }

    /**
     * Returns the Capabilities of this filter.
     *
     * @return the capabilities of this object
     * @see Capabilities
     */
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();

        // attributes
        //TODO: only string attributes
        result.enableAllAttributes();
        result.enable(Capability.MISSING_VALUES);

        // class
        result.enableAllClasses();
        result.enable(Capability.MISSING_CLASS_VALUES);
        result.enable(Capability.NO_CLASS);

        return result;
    }

    /**
     * Sets the format of the input instances.
     *
     * @param instanceInfo structure of the input instances
     *
     * @return true if the outputFormat may be collected immediately
     * @throws Exception
     *             if the input format can't be set successfully
     */
    public boolean setInputFormat(Instances instanceInfo) throws Exception {

        super.setInputFormat(instanceInfo);
        m_FirstBatchDone = false;

        if (!instanceInfo.attribute(0).isString() || instanceInfo.numAttributes() != 2) {
            throw new Exception("Illegal input format. Should be <string, class>.");
        }

        FastVector attInfo = new FastVector(m_AA.length() + 1);
        for (int i = 0; i < m_AA.length(); i++) {
            attInfo.addElement(new Attribute("a" + i));
        }
        Attribute labelAttr = (Attribute) getInputFormat().classAttribute().copy();
        attInfo.addElement(labelAttr);
        Instances outputFormat = new Instances(instanceInfo.relationName() + "-" + toString(), attInfo, 0);
        outputFormat.setClassIndex(outputFormat.numAttributes() - 1);
        setOutputFormat(outputFormat);

        return true;
    }

    /**
     * Input an instance for filtering.
     * @param instance  the input instance
     * @return true if the filtered instance may now be
     * collected with output().
     * @throws Exception if the input format was not set
     */
    public boolean input(Instance instance) {
        if (getInputFormat() == null) {
            throw new IllegalStateException("No input instance format defined");
        }
        if (getOutputFormat() == null) {
            // create new Instances
            FastVector attInfo = new FastVector(m_AA.length() + 1);
            for (int i = 0; i < m_AA.length(); i++) {
                attInfo.addElement(new Attribute("a" + i));
            }
            Attribute labelAttr = (Attribute) getInputFormat().classAttribute().copy();
            attInfo.addElement(labelAttr);
            Instances instances = new Instances(instance.dataset().relationName() + "-" + toString(), attInfo, 0);
            instances.setClassIndex(instances.numAttributes() - 1);
            setOutputFormat(instances);
        }

        loadWeights();

        if (m_NewBatch) {
            resetQueue();
            m_NewBatch = false;
        }
        convertInstance(instance);
        return true;
    }

    /**
     * Signify that this batch of input to the filter is finished. If the filter
     * requires all instances prior to filtering, output() may now be called to
     * retrieve the filtered instances.
     *
     * @return true if there are instances pending output
     * @throws IllegalStateException
     *             if no input structure has been defined
     */
    public boolean batchFinished() {
        if (getInputFormat() == null) {
            throw new IllegalStateException("No input instance format defined");
        }

        // Convert pending input instances
        for (int i = 0; i < getInputFormat().numInstances(); i++) {
            convertInstance(getInputFormat().instance(i));
        }

        // Free memory
        flushInput();

        m_NewBatch = true;
        return (numPendingOutput() != 0);
    }

    /**
     * Converts an input instance into output.
     * @param instance  The input instance.
     */
    private void convertInstance(Instance instance) {
        Instance tmp = new Instance(getOutputFormat().numAttributes());
        tmp.setDataset(getOutputFormat());
        String peptide = instance.stringValue(0);
        String s = compose(peptide);
        StringTokenizer st = new StringTokenizer(s, " ", false);
        int x = 0;
        while (st.hasMoreTokens()) {
            tmp.setValue(x++, Double.parseDouble(st.nextToken()));
        }
        tmp.setClassValue(instance.classValue());
        push(tmp);
    }

    /**
     * Compute aa compositions
     *
     * @param seq
     * @return
     */
    private String compose(String seq) {
        String val = new String();
        double len = seq.length();

        NumberFormat format = new DecimalFormat("##.##");
        for (int i = 0; i < m_AA.length(); i++) {
            int x = -1;
            double count = 0;
            while ((x = seq.indexOf(m_AA.substring(i, i + 1), x + 1)) >= 0) {
                count++;
            }
            val += " " + format.format((double) ((count * m_Weights[i]) / len));
        }

        return val.trim();
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
     *
     * @return A string description of this filter and its parameters.
     */

    public String toString() {
        return "SequenceComposition -S " + m_PropensityScale + " -A " + m_AA;
    }

    /**
     * Main method for testing this class.
     *
     * @param argv
     *
     */
    public static void main(String[] argv) {
        runFilter(new SequenceComposition(), argv);
    }
}
