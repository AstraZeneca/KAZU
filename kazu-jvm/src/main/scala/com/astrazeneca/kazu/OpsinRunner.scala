package com.astrazeneca.kazu

import uk.ac.cam.ch.wwmm.opsin.NameToStructure;
import uk.ac.cam.ch.wwmm.opsin.NameToStructureConfig;
import uk.ac.cam.ch.wwmm.opsin.OpsinResult;

import java.util
import scala.collection.JavaConverters._

class OpsinRunner {
  /**
   * wrapper for Opsin, returning a String object. This reduces cross process communication, as py4j only
   * needs to make one call per text string. JavaConverters are needed so that Py4J correctly assigns Py4J types/
   * unboxing is handled correctly
   */
  val n2sconfig = new NameToStructureConfig()
  val nts = NameToStructure.getInstance()
  val extendedSmiles = false

  def nameToStructure(name: String): String = {
    val result = nts.parseChemicalName(name, n2sconfig)
    val output = result.getExtendedSmiles()
    if (output == null)
      throw new Error(result.getMessage())
    output
  }
}
