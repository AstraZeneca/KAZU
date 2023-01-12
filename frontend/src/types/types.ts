type Entity = {
    match: string;
    start: number;
    end: number;
    entity_class: string;
}

type Section = {
    text: string;
    entities: Entity[];
}

type KazuWebDocument = {
    sections: Section[];
}

type RawKazuDocument = any;

type KazuNERResponse = {
    parsedDocument: KazuWebDocument;
    rawDocument: RawKazuDocument;
}

type KazuLSResponse = {
    ls_view: string,
    ls_tasks: object[],
    rawDocument: RawKazuDocument
}

export type {Entity, Section, KazuWebDocument, KazuNERResponse, KazuLSResponse}
