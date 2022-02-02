package com.astrazeneca.kazu
import org.junit.runner.RunWith
import org.scalatest.funsuite.AnyFunSuite
import org.scalatestplus.junit.JUnitRunner
import scala.collection.JavaConverters._

@RunWith(classOf[JUnitRunner])
class SethRunnerTest extends AnyFunSuite {
  test("SethRunner should find mutations") {
    val sethRunner: SethRunner = new SethRunner()
    val text = "Causative GJB2 mutations were identified in 31 (15.2%) patients, and two common mutations, c.35delG " +
      "and L90P (c.269T>C), accounted for 72.1% and 9.8% of GJB2 disease alleles."
    val result = sethRunner.findMutations(text).asScala
    assert(result.toList.size ==3)
  }
}
