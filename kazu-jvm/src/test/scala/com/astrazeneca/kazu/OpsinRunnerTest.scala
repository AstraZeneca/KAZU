package com.astrazeneca.kazu
import org.junit.runner.RunWith
import org.scalatest.funsuite.AnyFunSuite
import org.scalatestplus.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class OpsinRunnerTest extends AnyFunSuite {
  test("OpsinRunner should convert chemical names") {
    val opsinRunner: OpsinRunner = new OpsinRunner()
    val text = "2,2'-ethylenedipyridine"
    val result = opsinRunner.nameToStructure(text)
    assert(result=="C(CC1=NC=CC=C1)C1=NC=CC=C1")
  }
}
