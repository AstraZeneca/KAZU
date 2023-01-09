import {Entity, KazuWebDocument, Section} from "../types/types";
import * as R from "rambda";

class KazuUtils {
    public static entitiesFromDocument(kazuDocument: KazuWebDocument): Entity[] {
        const sections = kazuDocument.sections;
        return R.chain((section: Section) => section.entities) (sections);
    }

    public static textFromSimpleDocument(kazuDocument: KazuWebDocument): string {
        return kazuDocument.sections[0].text;
    }
}

export {KazuUtils}
