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
 *    CTD.java
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
 * <!-- globalinfo-start --> A filter for converting an amino acid sequence into numeric
 * features using the composition-distribution-transition (CTD) method described in:
 * EL-Manzalawy Y, Dobbs D, Honavar V (2008) On evaluating MHC-II binding peptide prediction
 * methods. PLoS ONE 3: e3268.
 *
 *  <p/> <!-- globalinfo-end -->
 *
 * @author Yasser EL-Manzalawy (yasser@cs.iastate.edu)
 * @version $Revision: 0.9 $
 */
public class CTD extends Filter implements UnsupervisedFilter {

    /** for serialization */
    static final long serialVersionUID = 3119607037607101160L;
    /** standard 20 amino acids */
    private String aa = "ACDEFGHIKLMNPQRSTVWY";
    /** grouping of amino acids into 3 groups using different physico-chemical properties */
    private String hydrophobicity = "23113223133121122332";
    private String polarizability = "11123132323212311233";
    private String polarity = "21331231311323322111";
    private String volume = "12123132323222311233";

    /**
     * Returns a string describing this classifier
     *
     * @return a description of the classifier suitable for displaying in the
     *         explorer/experimenter gui
     */
    public String globalInfo() {
        return "The the composition-distribution-transition (CTD) Filter implemented by Yasser EL-Manzalay \n" + "For more details, seee: EL-Manzalawy Y, Dobbs D, Honavar V (2008) On evaluating MHC-II binding peptide prediction" + " methods. PLoS ONE 3: e3268.";
    }

    /**
     * Returns the Capabilities of this filter.
     *
     * @return Filter capabilities.
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
     * @param instanceInfo
     *
     * @return true if the outputFormat may be collected immediately.
     * @throws Exception
     *             if the input format can't be set successfully.
     */
    public boolean setInputFormat(Instances instanceInfo) throws Exception {

        super.setInputFormat(instanceInfo);
        setOutputFormat(instanceInfo);
        m_FirstBatchDone = false;

        FastVector attInfo = new FastVector(104 + 1);
        for (int i = 0; i < 104; i++) {
            attInfo.addElement(new Attribute("a" + i));
        }
        Attribute labelAttr = (Attribute) getInputFormat().classAttribute().copy();
        attInfo.addElement(labelAttr);
        Instances outputFormat = new Instances("ctdData", attInfo, 0);
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
            FastVector attInfo = new FastVector(104 + 1);
            for (int i = 0; i < 104; i++) {
                attInfo.addElement(new Attribute("a" + i));
            }
            Attribute labelAttr = (Attribute) getInputFormat().classAttribute().copy();
            attInfo.addElement(labelAttr);
            Instances instances = new Instances("ctdData", attInfo, 0);
            instances.setClassIndex(instances.numAttributes() - 1);
            setOutputFormat(instances);
        }
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
        String s = compose(peptide) + " " + string2CDT(str2TriState(peptide, hydrophobicity)) + string2CDT(str2TriState(peptide, polarizability)) + string2CDT(str2TriState(peptide, polarity)) + string2CDT(str2TriState(peptide, volume));
        StringTokenizer st = new StringTokenizer(s, " ", false);
        int x = 0;
        while (st.hasMoreTokens()) {
            tmp.setValue(x++, Double.parseDouble(st.nextToken()));
        }
        tmp.setClassValue(instance.classValue());
        tmp.setWeight(instance.weight()); // keep the same weight of the
        // instance
        push(tmp);
    }

    /**
     * Converts an input peptide into a string defined on the alphabet {1,2,3}
     * @param peptide input peptide
     * @param property amino acid index to convert the peptide
     * @return converted string
     */
    private String str2TriState(String peptide, String property) {
        String s = new String();
        for (int i = 0; i < peptide.length(); i++) {
            int index = aa.indexOf(peptide.substring(i, i + 1));
            if (index >= 0) // skip other characters
            {
                s += property.substring(index, index + 1);
            }
        }
        return s;
    }

    /**
     * Converts an input peptide into CTD features
     * @param peptide
     * @return CTD features as a single space-separated String
     */
    private String string2CDT(String peptide) {
        int len = peptide.length();
        NumberFormat format = new DecimalFormat("##.##");

        // compositions
        double count1 = 0, count2 = 0, count3 = 0;
        int x = -1;
        while ((x = peptide.indexOf("1", x + 1)) >= 0) {
            count1++;
        }

        x = -1;
        while ((x = peptide.indexOf("2", x + 1)) >= 0) {
            count2++;
        }

        x = -1;
        while ((x = peptide.indexOf("3", x + 1)) >= 0) {
            count3++;
        }

        // Transitions
        double T1 = 0, T2 = 0, T3 = 0;

        x = -1;
        while ((x = peptide.indexOf("12", x + 1)) >= 0) {
            T1++;
        }
        x = -1;
        while ((x = peptide.indexOf("21", x + 1)) >= 0) {
            T1++;
        }

        x = -1;
        while ((x = peptide.indexOf("13", x + 1)) >= 0) {
            T2++;
        }
        x = -1;
        while ((x = peptide.indexOf("31", x + 1)) >= 0) {
            T2++;
        }

        x = -1;
        while ((x = peptide.indexOf("23", x + 1)) >= 0) {
            T3++;
        }
        x = -1;
        while ((x = peptide.indexOf("32", x + 1)) >= 0) {
            T3++;
        }

        // distributions
        // first, 25, 50, 75, 100

        // 1
        String D1 = new String();
        double pos = peptide.indexOf("1") + 1;
        D1 += format.format((pos / len) * 100);

        int index = (int) (count1 * 0.25);
        pos = 0;
        x = -1;
        while ((x = peptide.indexOf("1", x + 1)) >= 0) {
            pos++;
            if (pos >= index) {
                break;
            }
        }
        pos = x + 1;
        D1 += " " + format.format((pos / len) * 100);

        index = (int) (count1 * 0.5);
        pos = 0;
        x = -1;
        while ((x = peptide.indexOf("1", x + 1)) >= 0) {
            pos++;
            if (pos >= index) {
                break;
            }
        }
        pos = x + 1;
        D1 += " " + format.format((pos / len) * 100);

        index = (int) (count1 * 0.75);
        pos = 0;
        x = -1;
        while ((x = peptide.indexOf("1", x + 1)) >= 0) {
            pos++;
            if (pos >= index) {
                break;
            }
        }
        pos = x + 1;
        D1 += " " + format.format((pos / len) * 100);

        index = (int) (count1);
        pos = 0;
        x = -1;
        while ((x = peptide.indexOf("1", x + 1)) >= 0) {
            pos = x;
        }
        pos++;
        D1 += " " + format.format((pos / len) * 100);

        // 2
        String D2 = new String();
        pos = peptide.indexOf("2") + 1;
        D2 += format.format((pos / len) * 100);

        index = (int) (count2 * 0.25);
        pos = 0;
        x = -1;
        while ((x = peptide.indexOf("2", x + 1)) >= 0) {
            pos++;
            if (pos >= index) {
                break;
            }
        }
        pos = x + 1;
        D2 += " " + format.format((pos / len) * 100);

        index = (int) (count2 * 0.5);
        pos = 0;
        x = -1;
        while ((x = peptide.indexOf("2", x + 1)) >= 0) {
            pos++;
            if (pos >= index) {
                break;
            }
        }
        pos = x + 1;
        D2 += " " + format.format((pos / len) * 100);

        index = (int) (count2 * 0.75);
        pos = 0;
        x = -1;
        while ((x = peptide.indexOf("2", x + 1)) >= 0) {
            pos++;
            if (pos >= index) {
                break;
            }
        }
        pos = x + 1;
        D2 += " " + format.format((pos / len) * 100);

        index = (int) (count2);
        pos = 0;
        x = -1;
        while ((x = peptide.indexOf("2", x + 1)) >= 0) {
            pos = x;
        }
        pos++;
        D2 += " " + format.format((pos / len) * 100);

        // 3
        String D3 = new String();
        pos = peptide.indexOf("3") + 1;
        D3 += format.format((pos / len) * 100);

        index = (int) (count3 * 0.25);
        pos = 0;
        x = -1;
        while ((x = peptide.indexOf("3", x + 1)) >= 0) {
            pos++;
            if (pos >= index) {
                break;
            }
        }
        pos = x + 1;
        D3 += " " + format.format((pos / len) * 100);

        index = (int) (count3 * 0.5);
        pos = 0;
        x = -1;
        while ((x = peptide.indexOf("3", x + 1)) >= 0) {
            pos++;
            if (pos >= index) {
                break;
            }
        }
        pos = x + 1;
        D3 += " " + format.format((pos / len) * 100);

        index = (int) (count3 * 0.75);
        pos = 0;
        x = -1;
        while ((x = peptide.indexOf("3", x + 1)) >= 0) {
            pos++;
            if (pos >= index) {
                break;
            }
        }
        pos = x + 1;
        D3 += " " + format.format((pos / len) * 100);

        index = (int) (count3);
        pos = 0;
        x = -1;
        while ((x = peptide.indexOf("3", x + 1)) >= 0) {
            pos = x;
        }
        pos++;
        D3 += " " + format.format((pos / len) * 100);

        return format.format((count1 / len) * 100) + " " + format.format((count2 / len) * 100) + " " + format.format((count3 / len) * 100) + " " + format.format((T1 / (len - 1)) * 100) + " " + format.format((T2 / (len - 1)) * 100) + " " + format.format((T3 / (len - 1)) * 100) + " " + D1 + " " + D2 + " " + D3 + " ";
    }

    /**
     * compute aa compositions
     *
     * @param seq
     * @return
     */
    private String compose(String seq) {
        String val = new String();
        double len = seq.length();

        NumberFormat format = new DecimalFormat("##.##");
        for (int i = 0; i < aa.length(); i++) {
            int x = -1;
            double count = 0;
            while ((x = seq.indexOf(aa.substring(i, i + 1), x + 1)) >= 0) {
                count++;
            }
            val += " " + format.format((double) (count / len));
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
     * Main method for testing this class.
     *
     * @param argv
     *            should contain arguments to the filter: use -h for help
     */
    public static void main(String[] argv) {
        runFilter(new CTD(), argv);
    }
}
