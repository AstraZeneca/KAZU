import {KazuClient, IKazuClient} from "./kazu-client";

const itif = (condition: boolean) => {
    if(condition){
        return it;
    } else {
        return it.skip;
    }
}


describe("Kazu client tests", () => {
    const kazuApiUrl: string|undefined = process.env.KAZU_API_URL;

    itif(kazuApiUrl !== undefined)("should parse kazu responses", () => {
        const kazu_client = new KazuClient(kazuApiUrl as string)
        kazu_client.ner("EGFR is a gene implicated in breast cancer")
            .then(kazu_web_response => {
                expect(kazu_web_response.parsedDocument.sections.length).toBeGreaterThan(0)
            })
    })
});
