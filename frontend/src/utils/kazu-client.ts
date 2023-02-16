import {KazuLSResponse} from "../types/types";
import axios from "axios";

interface IKazuClient {
    ner_with_ls(text: string, auth?: string): Promise<KazuLSResponse>
}

class KazuClient implements IKazuClient {
    private kazuApiUrl: string;

    constructor(kazuApiUrl: string) {
        this.kazuApiUrl = kazuApiUrl
    }

    ner_with_ls(text: string, auth?: string | undefined): Promise<KazuLSResponse> {
        const basicReq: any = {
            url: `${this.kazuApiUrl}/api/kazu/ls-annotations`,
            data: {text: text},
            method: "POST"
        }

        const req: any = auth !== undefined ? {...basicReq, headers: {"Authorization": `Bearer ${auth}`}} : basicReq;

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
export type {IKazuClient};
