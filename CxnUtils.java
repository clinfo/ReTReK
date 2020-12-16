import chemaxon.checkers.ValenceErrorChecker;
import chemaxon.checkers.result.StructureCheckerResult;
import chemaxon.formats.MolExporter;
import chemaxon.formats.MolFormatException;
import chemaxon.formats.MolImporter;
import chemaxon.reaction.ReactionException;
import chemaxon.reaction.Reactor;
import chemaxon.sss.SearchConstants;
import chemaxon.sss.search.SearchOptions;
import chemaxon.struc.Molecule;
import chemaxon.struc.RxnMolecule;
// import org.nd4j.linalg.io.CollectionUtils;
import org.apache.commons.collections4.CollectionUtils;
import py4j.GatewayServer;

import java.io.IOException;
import java.io.File;
import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class CxnUtils {
    public static List<RxnMolecule> rolloutList;

    public static List<List<String>> reactProductToReactants(String strPro, String strRxnRule) throws IOException{
        Molecule[] reactantsArray = null;
        Molecule product = null;
        List<Molecule> reactantsList = null;
        List<List<String>> returnList = new ArrayList<List<String>>();

        SearchOptions searchOptions = new SearchOptions(SearchConstants.DEFAULT_SEARCHTYPE);
        searchOptions.setStereoModel(SearchOptions.STEREO_MODEL_LOCAL);
        searchOptions.setStereoSearchType(SearchOptions.STEREO_EXACT);

        Reactor reactor = new Reactor();
        try {
            product = stringToMolecule(strPro);
            Molecule[] prd = {product};
            reactor.setReverse(true);
            reactor.setReactionString(strRxnRule);
            reactor.setReactants(prd);
            reactor.setSearchOptions(searchOptions.toString());
            while ((reactantsArray = reactor.react()) != null) {
                reactantsList = new ArrayList<>();
                reactantsList.addAll(Arrays.asList(reactantsArray));
                if (!isValidReactionResult(product, reactantsArray, strRxnRule)) {
                    continue;
                }
                for (Molecule mol : reactantsList) {
                    if (!isValidValence(mol)) {
                        continue;
                    }
                }
                if (CollectionUtils.isEmpty(reactantsList)) {
                    continue;
                } else {
                    List<String> tmpList = new ArrayList<>();
                    for (Molecule r : reactantsList) {
                        tmpList.add(MolExporter.exportToFormat(r, "smiles"));
                    }
                    returnList.add(tmpList);
                }
            }

        } catch (ReactionException | MolFormatException e) {
            e.printStackTrace();
            System.exit(1);
        }
        return CollectionUtils.isEmpty(reactantsList) ? null : returnList;
    }

    static List<Molecule> reactProductToReactants(Molecule product, RxnMolecule reactionRule) throws ReactionException {
        Molecule[] reactantsArray;
        List<Molecule> reactantsList = new ArrayList<>();
        SearchOptions searchOptions = new SearchOptions(SearchConstants.DEFAULT_SEARCHTYPE);
        searchOptions.setStereoModel(SearchOptions.STEREO_MODEL_LOCAL);
        searchOptions.setStereoSearchType(SearchOptions.STEREO_EXACT);

        Reactor reactor = new Reactor();
        Molecule[] prd = {product};
        reactor.setReverse(true);
        reactor.setReaction(reactionRule);
        reactor.setReactants(prd);
        reactor.setSearchOptions(searchOptions.toString());
        reactantsArray = reactor.react();
        if (reactantsArray != null) {
            reactantsList.addAll(Arrays.asList(reactantsArray));
        }
        for (Molecule mol : reactantsList) {
            if (!isValidValence(mol)) {
                return null;
            }
        }
        return CollectionUtils.isEmpty(reactantsList) ? null : reactantsList;
    }

    public static Molecule stringToMolecule(String strMol) throws MolFormatException {
        return MolImporter.importMol(strMol);
    }

    public static Boolean isValidValence(Molecule mol) {
        ValenceErrorChecker checker = new ValenceErrorChecker();
        StructureCheckerResult result = checker.check(mol);
        return result == null;
    }

    static Boolean isValidReactionResult(Molecule product, Molecule[] reactants, String rxnTemplate) {
        SearchOptions searchOptions = new SearchOptions(SearchConstants.DEFAULT_SEARCHTYPE);
        searchOptions.setStereoModel(SearchOptions.STEREO_MODEL_LOCAL);
        searchOptions.setStereoSearchType(SearchOptions.STEREO_EXACT);
        String proUniq;
        String genProductUniq;
        Molecule[] genProduct;
        Reactor reactor;
        try {
            proUniq = MolExporter.exportToFormat(product, "smiles:u");
        } catch (IOException e) {
            return false;
        }
        try {
            reactor = new Reactor();
            reactor.setReverse(false);
            reactor.setReactionString(rxnTemplate);
            reactor.setReactants(reactants);
            reactor.setSearchOptions(searchOptions.toString());
            while ((genProduct = reactor.react()) != null) {
                if (genProduct.length != 1) {
                    continue;
                }
                try {
                    genProductUniq = MolExporter.exportToFormat(genProduct[0], "smiles:u");
                } catch (IOException e) {
                    continue;
                }
                if (genProductUniq.equals(proUniq)) {
                    return true;
                }
            }
        } catch (ReactionException e) {
            return false;
        }
        return false;
    }

    static List<RxnMolecule> getReactionList(String reactionRuleListPath) throws IOException {
        MolImporter importer = new MolImporter(new FileInputStream(new File(reactionRuleListPath)));
        Molecule m;
        List<RxnMolecule> rxnRuleList = new ArrayList<>();
        while ((m = importer.read()) != null) {
            RxnMolecule r = RxnMolecule.getReaction(m);
            rxnRuleList.add(r);
        }
        return rxnRuleList;
    }

    static List<Molecule> getMolList(List<String> mols) {
        List<Molecule> molList = new ArrayList<>();
        try {
            for (String strMol : mols) {
                Molecule mol = MolImporter.importMol(strMol);
                molList.add(mol);
            }
        } catch (MolFormatException e) {
            e.printStackTrace();
        }
        return molList;
    }

    public static Boolean isTerminal(ArrayList<String> strMolList) throws ReactionException {
        List<Molecule> molList = getMolList(strMolList);
        List<Molecule> reactantsList = null;
        for (Molecule mol : molList) {
            for (RxnMolecule rxn : CxnUtils.rolloutList) {
                try {
                    reactantsList = reactProductToReactants(mol, rxn);
                } catch (ReactionException ignore) {}
                if (reactantsList != null) {
                    return false;
                }
            }
        }
        return true;
    }

    public static void main(String[] args) {
        //List<RxnMolecule> reactionList = new ArrayList<>();
        try {
             CxnUtils.rolloutList = getReactionList(args[1]);
        } catch (IOException e) {
            e.printStackTrace();
        }
        CxnUtils cu = new CxnUtils();
        int port = Integer.parseInt(args[0]);
        GatewayServer server = new GatewayServer(cu, port);
        server.start();
        System.out.println("A gateway server started");
    }
}
