import {Entity, KazuLSResponse, KazuNERResponse, KazuWebDocument, Section} from "../types/types";
import axios from "axios";
import * as R from "rambda";

class JsonParseError extends Error {
    constructor(message: string, from_error: Error) {
        super(message, {cause: from_error});
    }
}
interface IKazuClient {
    ner(text: string, auth?: string): Promise<KazuNERResponse>
    ner_with_ls(text: string, auth?: string): Promise<KazuLSResponse>
}
class KazuClient implements IKazuClient {
    private kazuApiUrl: string;
    constructor(kazuApiUrl: string) {
       this.kazuApiUrl = kazuApiUrl
    }

    private static parseKazuEntity(kazuEntity: any): Entity {
        return {
            match: kazuEntity["match"],
            start: kazuEntity["start"],
            end: kazuEntity["end"],
            entity_class: kazuEntity["entity_class"]
        }
    }

    private static parseKazuSection(kazuSection: any): Section {
        const entitiesJson = "entities" in kazuSection? kazuSection["entities"] as Array<object> : [];
        return {
            text: kazuSection["text"],
            entities: R.map((entityJson: object) => KazuClient.parseKazuEntity(entityJson))(entitiesJson)
        }
    }

    private static parseKazuWebDocument(kazuWebDocument: any): KazuWebDocument {
        const sectionsJson = kazuWebDocument["sections"] as Array<any>;
        return {
            sections: R.map((sectionJson: any) => KazuClient.parseKazuSection(sectionJson))(sectionsJson)
        }
    }

    private static parseKazuResponse(kazuResponse: any): KazuWebDocument {
        try {
            return KazuClient.parseKazuWebDocument(kazuResponse.data);
        } catch (e: any) {
            throw new JsonParseError("Failed to parse Kazu web response", e);
        }
    }
    ner(text: string, auth?: string): Promise<KazuNERResponse> {
        let req;
        if(auth === undefined) {
            req = {
                url: `${this.kazuApiUrl}/api/kazu`,
                data: {text: text},
                method: "POST"
            }
        } else {
            req = {
                url: `${this.kazuApiUrl}/api/kazu`,
                data: {text: text},
                method: "POST",
                headers: {
                    "Authorization": `Bearer ${auth}`
                }
            }
        }
        return axios(req)
            .then((kazuWebResponseJson: any) => {
                return {
                    parsedDocument: KazuClient.parseKazuResponse(kazuWebResponseJson),
                    rawDocument: kazuWebResponseJson.data
                }
            })
    }

    ner_with_ls(text: string, auth?: string | undefined): Promise<KazuLSResponse> {
        let req;
        if(auth === undefined) {
            req = {
                url: `${this.kazuApiUrl}/api/kazu/ls-annotations`,
                data: {text: text},
                method: "POST"
            }
        } else {
            req = {
                url: `${this.kazuApiUrl}/api/kazu/ls-annotations`,
                data: {text: text},
                method: "POST",
                headers: {
                    "Authorization": `Bearer ${auth}`
                }
            }
        }

        return axios(req)
            .then((resp: any) => {
                const respData = resp.data
                const lsView = respData["ls_view"] as string
                const lsTasks = respData["ls_tasks"]
                const rawDocument = respData["doc"]

                return {
                    ls_view: lsView,
                    ls_tasks: lsTasks,
                    rawDocument: rawDocument
                }
            })
    }
}

export {KazuClient};
export type { IKazuClient };
