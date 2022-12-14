package com.astrazeneca.kazu

import de.hu.berlin.wbi.objects.MutationMention
import seth.SETH

import java.util
import scala.collection.JavaConverters._

class SethRunner {
  /**
   * wrapper for SETH, returning a json like Map object. This reduces cross process communication, as py4j only
   * needs to make one call per text string. JavaConverters are needed so that Py4J correctly assigns Py4J types/
   * unboxing is handled correctly
   */
  val annotator = new SETH()

  def findMutations(text: String): util.List[util.Map[String, Any]] = {
    annotator.findMutations(text).asScala.map { mutationMention: MutationMention =>
      Map(
        "start" -> mutationMention.getStart,
        "end" -> mutationMention.getEnd,
        "text" -> mutationMention.getText,
        "hgvs" -> mutationMention.toHGVS,
        "wtResidue" -> mutationMention.getWtResidue,
        "mutResidue" -> mutationMention.getMutResidue,
        "mutation_type" -> mutationMention.getType.toString,
        "found_with" -> mutationMention.getTool.toString,
        "protein_mutation" -> mutationMention.isPsm,
        "nucleotide_mutation" -> mutationMention.isNsm,
        "ambiguous"-> mutationMention.isAmbiguous
      ).asJava
    }.toList.asJava
  }
}
