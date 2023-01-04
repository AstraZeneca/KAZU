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

type KazuResponse = {
    parsedDocument: KazuWebDocument;
    rawDocument: RawKazuDocument;
}

export type {Entity, Section, KazuWebDocument, KazuResponse}