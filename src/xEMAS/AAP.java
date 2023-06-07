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
 *    AAP.java
 *    Copyright (C) 2009 Yasser EL-Manzalawy
 *
 */
package epit.filters.unsupervised.attribute;

import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.filters.Filter;
import weka.filters.UnsupervisedFilter;
import java.util.*;

/**
 * <!-- globalinfo-start --> A filter for converting an amino acid sequence into 400 numeric 
 * features using the amino acid pairs propensity scale method. For more details see:
 * Chen J, Liu H, Yang J, Chou K (2007) Prediction of linear B-cell epitopes using amino 
 * acid pair antigenicity scale. Amino Acids 33: 423-428.
 * <p/> <!-- globalinfo-end  -->
 *
 * @author Yasser EL-Manzalawy (yasser@cs.iastate.edu)
 * @version $Revision: 0.9 $
 * @see SequenceDiCompositions
 */
public class AAP extends Filter implements UnsupervisedFilter {

    /** for serialization */
    static final long serialVersionUID = 3119607037607101160L;
    /** Amino acid dipeptides  */
    private String m_AADi = "AA AC AD AE AF AG AH AI AK AL AM AN AP AQ AR AS AT AV AW AY CA CC CD CE CF CG CH CI CK CL CM CN CP CQ CR CS CT CV CW CY DA DC DD DE DF DG DH DI DK DL DM DN DP DQ DR DS DT DV DW DY EA EC ED EE EF EG EH EI EK EL EM EN EP EQ ER ES ET EV EW EY FA FC FD FE FF FG FH FI FK FL FM FN FP FQ FR FS FT FV FW FY GA GC GD GE GF GG GH GI GK GL GM GN GP GQ GR GS GT GV GW GY HA HC HD HE HF HG HH HI HK HL HM HN HP HQ HR HS HT HV HW HY IA IC ID IE IF IG IH II IK IL IM IN IP IQ IR IS IT IV IW IY KA KC KD KE KF KG KH KI KK KL KM KN KP KQ KR KS KT KV KW KY LA LC LD LE LF LG LH LI LK LL LM LN LP LQ LR LS LT LV LW LY MA MC MD ME MF MG MH MI MK ML MM MN MP MQ MR MS MT MV MW MY NA NC ND NE NF NG NH NI NK NL NM NN NP NQ NR NS NT NV NW NY PA PC PD PE PF PG PH PI PK PL PM PN PP PQ PR PS PT PV PW PY QA QC QD QE QF QG QH QI QK QL QM QN QP QQ QR QS QT QV QW QY RA RC RD RE RF RG RH RI RK RL RM RN RP RQ RR RS RT RV RW RY SA SC SD SE SF SG SH SI SK SL SM SN SP SQ SR SS ST SV SW SY TA TC TD TE TF TG TH TI TK TL TM TN TP TQ TR TS TT TV TW TY VA VC VD VE VF VG VH VI VK VL VM VN VP VQ VR VS VT VV VW VY WA WC WD WE WF WG WH WI WK WL WM WN WP WQ WR WS WT WV WW WY YA YC YD YE YF YG YH YI YK YL YM YN YP YQ YR YS YT YV YW YY";
    /** Amino acid  dipeptides' scale values */
    private String m_AAP_scale = "-0.10654021308331996 -0.09922775190833943 -0.047542219408659325 0.14048180802388543 0.43650542728281727 0.014220672556194414 -0.29958002390779725 -0.40985065453429115 -0.005972115615230367 -0.45911441040892387 -0.05123090483659487 0.2449664279225634 0.29601427495096067 -0.02365478434550139 -0.05503590069768338 0.08788448595990306 -8.389092763365635E-4 0.013863650663772153 0.20335563346401764 -0.08471378673731478 0.1845123968026876 0.12693094487031753 -0.647996437481924 -0.3567898911981128 -0.4972377699523316 -0.2268173884059742 0.14202971687441912 0.09782388324992253 0.00605151311473362 -0.3441588043907119 -0.41387793358376535 0.46339376285248823 -0.18810592303189688 -0.005751247786846769 -0.14852836370353462 0.5847513537355498 0.8195164239226997 0.4076541998801475 0.5083497541526831 0.061567069848009304 -0.15468380980990515 0.12586471146755862 0.07450927276642494 -0.08850752419595942 -0.3383112070370726 0.15321707363979398 0.19433270278343473 -0.1249514155802468 0.2745518915944274 -0.031786689255788314 0.17856209297636072 0.033146962535795854 0.3363740471742651 0.1495505746537602 0.32139216697115036 0.03495076981465983 0.07650566908623624 -0.2602920681662432 0.26431257219781346 0.0820763560545712 -0.04102635413549016 -0.14424309789874523 0.05344621246758985 0.28270478944541577 -0.19846487593194195 0.13420075002568033 0.1388686120704421 -0.0732352360698243 0.10512920939896753 -0.029074555979456185 -0.12167882365776972 0.10467194006754199 0.3723651148258895 0.21679743246804706 0.20940058300901399 0.13258008868695348 -0.09744843133622383 0.05009401579237216 0.4009013906217356 0.31177983975554047 -0.18962817482701422 0.13368366372947182 -0.07162119305369974 -0.2927079987815693 -0.1792529817041406 -0.1295201595466292 0.02564667192617831 -0.7045702498039834 -0.08739450797743786 -0.3127579521331201 -0.41769751560232093 -0.11513745793597296 -0.07145628617535815 -0.46493419656834156 0.03672587021455587 -0.49985396507316227 -0.21853144445177697 0.08133452448577483 -0.37863076893392333 0.4778795252696919 0.11496713242182977 0.6169485944644497 0.21667810719449676 0.008887077073335803 0.1117000646595625 0.24780364165329227 -0.22937178757024212 0.17524554605284193 0.3419541333393441 -0.30465305118728225 -0.07760261215934094 0.3173665037721134 0.8204589721245994 0.05391128526617872 0.4455081428368035 -0.09449053847453115 0.08267065447703659 -0.084623649461122 0.22616761187473644 -0.25641935787060455 -0.16548996890451972 -0.14262436637279696 -0.32342764387416223 0.028951325571359154 -0.4828119107843887 0.09186307521369952 -0.018579460751047883 0.3163336344671466 0.19650175826667304 -0.34560350705079046 -0.500029072794741 0.1455681697689004 0.2386785230536035 0.12970701772408066 -0.08839084910028538 0.07949259777930506 0.022679708203354965 0.06482974698356081 0.6191420718190306 0.12037321577431892 -0.3178163243199583 0.25160091325087564 -0.29243945609950484 0.0010461410052189368 -0.4263566356948908 0.22501099092389554 0.35371097408581464 0.013037873800904043 -0.20138513983838724 -0.12236947686559752 -0.5395971618689811 0.08479964145784824 0.12612623510029186 0.010267712699974174 0.32861159973082854 -0.21885412671471804 -0.049288837531081486 -0.1533731682548075 0.7604558193111004 -0.11811985469781983 0.10608449572133738 0.16164589856741451 0.15150614903662407 0.17877597387278987 -0.3609405031561922 0.15564752227015832 0.09524483754054525 0.18314417461907162 0.08828696033253691 0.05298000375399914 -0.44320360327994013 0.12899723788335948 0.2828938789691451 0.032703469836376176 0.21228055828564085 0.2818437803428919 0.020966039116591784 -0.07793442228866165 0.5070971971892158 0.021851024466356117 -0.24360675764401363 -0.20918200777426976 0.012524324402663822 -0.035690931605871445 -0.5554738886830015 -0.05676333059290517 -0.5268488027738607 -0.24135791262742345 0.055661136419318025 -0.2435541801765716 -0.40490736982204767 -0.2865985051246964 -0.15679553273310776 -0.13672150520463722 -0.1572142795113477 -0.4124579062986048 -0.12301488603770216 -0.29741383419278167 0.34970763718927445 -0.10318973315586932 -0.4409321306640275 -0.8897528319448026 -0.3439759845004926 -0.23329288617154265 -0.2801412046277695 -0.37265556615742124 0.19711314955678594 -0.7164040950646373 -0.26365902777201466 -0.07171151583495894 0.08014253824636364 -1.0 -0.6238989993171473 -0.0788956622646132 0.00850758398367657 -0.2190555672082407 0.15586219900285703 -0.20497732916154765 1.0 0.365853540666603 0.3363719557443694 0.8353207793023965 0.0578631692703675 0.1883640479375186 0.06229841092901034 0.03127672919548985 -0.1970340476620075 -0.20581968719380117 0.14845850506227043 -0.27294025475785877 0.23793148052632174 0.41477408193738596 0.3325652855331984 0.21901816116642436 -0.001265362517592017 0.009917768470722566 0.34141447284903714 0.10843538372510775 0.7385482138070874 0.09525362754500177 0.30114282967409256 0.12531541629711773 0.2605407586665196 0.17529399422543257 -0.014375970794022863 0.6330937407355146 -0.028422124086630585 0.1738432868890991 0.20411578645710748 0.015282771863843525 -0.3162232826996074 0.5045320756568836 0.4122476453607278 0.09224925880081436 0.145669575296697 -0.0790859218369494 0.3770872624420041 0.27201692276363865 0.5335458514453659 0.38648062605656497 -0.1259382883793736 -0.4521726678823673 -0.1374741017279847 0.32252015435065795 -0.29173945156602243 0.16059216814114396 0.06073760937100747 -0.03890729512275837 -0.10882147216835059 -0.047866354030631886 0.19246437672715433 0.18291937777560663 0.19371351530780556 0.1451150949462663 0.322056237373501 -0.22777254464426278 -0.3771899997056709 -0.07906634128772727 0.1490182866733536 0.22479632236967606 0.4662949474401783 -0.24062788570538463 0.23060346341935634 0.20302976161650532 -0.04012680324302087 0.3870654938529565 -0.13478303372458522 0.2605812069198292 0.26179842027020905 -0.2973775026805058 -0.06361627872103714 0.05999024225599303 0.17838916506125546 0.12575481187406923 0.16163584283529375 -0.13371281634773136 -0.04077819190483445 0.00306231396976564 0.38008777499469315 0.15158998029523074 -0.0804861168986214 0.19936542259563717 0.13450052671151003 -0.13690707648729794 -0.36868076984198384 0.11734031138275203 -0.31727434074951133 0.017763376931577923 -0.10558740549536094 -0.1618930414954891 -0.23720372732619865 -0.03769150350756845 0.08305561913903658 0.14646084417786276 -0.23823393642956492 -0.05009653147730053 -0.04917850843535421 -0.08915841206492214 0.49605145202396783 -0.3452360206070828 -0.027802838398411023 0.20147111749899138 0.06673306403423718 0.08897354493027199 -0.050145178664051104 0.18840796442020946 0.17574730778222714 0.17504476194802354 0.21753335817167851 -0.06653475818900367 -0.10011333672390021 0.274698039917608 0.13769059748078138 -0.04662952016602018 0.2058513808299196 -0.016190140624229854 0.37530835678113506 -0.30599190347140126 -0.19669680075639662 0.08626685790688104 -0.20102181064742097 -0.07648631198764988 -0.06933935130594304 0.03458070772335975 -0.5366120097085314 0.16163665191090426 0.30527480644817406 -0.3045241623512822 0.07509029646180898 -0.2662048118140714 -0.5312177552958843 -0.0751122821410306 0.15158069077163883 0.0685778640254755 -0.10682742610692642 -0.1872691982377439 0.19818991876815262 0.06570855447254953 0.3273571218951019 0.17508988952384041 0.34681850317050533 0.7081635173291885 0.6031227125230243 0.4014040419263216 0.060439286325327135 0.7880342102312741 -0.20094998406295517 -0.2291559819588871 0.6799225197900327 -0.2999291814864733 0.8151000377668645 0.5814196258174755 0.5085727071494166 0.42948384362799175 0.6424033277742967 8.814265624716988E-4 0.07437899063745812 0.28009064310256737 0.03414229154892201 0.2848814491747029 0.2736646527183624 0.14157076767105314 0.25453083394083653 0.009802160514739366 -0.28403359461794164 -0.14843246174422264 0.04288103367446139 0.18695200923228028 0.38748006388338463 -0.08169753590776618 -0.15747929461158938 0.08109819272831498 0.3730606767242801 -0.14544238358246075 0.003582446650076898 -0.08174572369247879 0.3065918218608654 -0.28387859766624124 -0.08894827384878312 0.1891303752387734";
    /** Dipeptide scales */
    private double[] scale;
    /** Amino acid dipeptides */
    private Vector m_AAPStr;

    /**
     * Returns a string describing this classifier
     *
     * @return a description of the classifier suitable for displaying in the
     *         explorer/experimenter gui
     */
    public String globalInfo() {
        return "Amino acids propensity scales (AAP) filter by Yasser EL-Manzalawy.\n" + " For more details about AAP method, see: " + "Chen J, Liu H, Yang J, Chou K (2007) Prediction of linear B-cell epitopes using " + "amino acid pair antigenicity scale. Amino Acids 33: 423-428. ";
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
     * @param instanceInfo structure of the input instances.
     *
     * @return true if the outputFormat may be collected immediately.
     * @throws Exception
     *             if the input format can't be set successfully.
     */
    public boolean setInputFormat(Instances instanceInfo) throws Exception {

        super.setInputFormat(instanceInfo);
        setOutputFormat(instanceInfo);
        m_FirstBatchDone = false;

        init();
        FastVector attInfo = new FastVector(400 + 1);
        for (int i = 0; i < 400; i++) {
            attInfo.addElement(new Attribute("a" + i));
        }
        Attribute labelAttr = (Attribute) getInputFormat().classAttribute().copy();
        attInfo.addElement(labelAttr);
        Instances outputFormat = new Instances(instanceInfo.relationName() + "-AAP", attInfo, 0);
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
    public boolean input(Instance instance) throws Exception {
        if (getInputFormat() == null) {
            throw new IllegalStateException("No input instance format defined");
        }
        if (getOutputFormat() == null) {
            // create new Instances
            FastVector attInfo = new FastVector(400 + 1);
            for (int i = 0; i < 400; i++) {
                attInfo.addElement(new Attribute("a" + i));
            }
            Attribute labelAttr = (Attribute) getInputFormat().classAttribute().copy();
            attInfo.addElement(labelAttr);
            Instances instances = new Instances(instance.dataset().relationName() + "-AAP", attInfo, 0);
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
     * Initializes the APP object.
     */
    private void init() {
        m_AAPStr = new Vector(); // e.g. AA, AC,...,YY
        // load AAP scale
        StringTokenizer st = new StringTokenizer(m_AADi, " ", false);
        while (st.hasMoreTokens()) {
            m_AAPStr.addElement(st.nextToken());
        }
        st = new StringTokenizer(m_AAP_scale, " ", false);
        int len = 400;
        scale = new double[len];
        for (int i = 0; i < len; i++) {
            scale[i] = Double.parseDouble(st.nextToken());
        }

    }

    /**
     * Converts an input instance into output.
     * @param instance input instance
     */
    private void convertInstance(Instance instance) {
        int len = 400;
        String s = instance.stringValue(0);
        double[] f = new double[len];
        for (int j = 0; j < len; j++) {
            f[j] = 0;
        }
        for (int j = 0; j < len; j++) {
            String t = (String) m_AAPStr.elementAt(j);
            int x = -1;
            while ((x = s.indexOf(t, x + 1)) >= 0) {
                f[j]++;
            }
        }
        Instance tmp = new Instance(len + 1);
        tmp.setDataset(getOutputFormat());
        int x = 0;
        for (int j = 0; j < len; j++) {
            tmp.setValue(x++, f[j] * scale[j]);
        }
        tmp.setClassValue(instance.classValue());
        push(tmp);
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
        runFilter(new AAP(), argv);
    }
}
