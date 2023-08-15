package com.astrazeneca.kazu

import uk.ac.cam.ch.wwmm.opsin.NameToStructure;
import uk.ac.cam.ch.wwmm.opsin.NameToStructureConfig;

class OpsinRunner {
  /* Wrapper for Opsin, returning a String object. */
  val n2sconfig = new NameToStructureConfig()
  val nts = NameToStructure.getInstance()

  def nameToStructure(name: String): String = {
    val result = nts.parseChemicalName(name, n2sconfig)
    val output = result.getExtendedSmiles()
    if (output == null)
      throw new Error(result.getMessage())
    output
  }
}
