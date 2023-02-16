import {KazuClient, IKazuClient} from "./kazu-client";

const testif = (condition: boolean) => {
    if(condition){
        return test;
    } else {
        return test.skip;
    }
}


describe("Kazu client tests", () => {
    const kazuApiUrl: string|undefined = process.env.KAZU_API_URL;

    itif(kazuApiUrl !== undefined)("should parse kazu responses", () => {
        const kazu_client = new KazuClient(kazuApiUrl as string)
        kazu_client.ner_with_ls("EGFR is a gene implicated in breast cancer")
            .then(kazu_web_response => {
                expect(kazu_web_response.rawDocument["sections"].length).toBeGreaterThan(0)
            })
    })
});
