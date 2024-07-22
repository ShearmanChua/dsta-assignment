import pydash
from assignment_1.assignment_1_1.convert_scanned_pdf import convert_scanned_pdf
from assignment_1.assignment_1_1.convert_epdf import convert_e_pdf
from assignment_1.assignment_1_1.data_chunk import create_document_string
from assignment_1.assignment_1_2.hybrid_db import setup_hybrid_db
from assignment_1.assignment_1_2.milvus_db import create_embedding, MilvusDB, setup_milvus_db
from langchain.schema.document import Document
from env import MILVUS_HOST, MILVUS_PORT,  MILVUS_DATABASE, ES_URL
from connexion.exceptions import BadRequestProblem


def build_document_index(pdf_bin, table_name, converter_engine, store_in='hybrid', 
                         milvus_openai_embedding_enabled=True, embedding_name='BgeEmbeddings', overwrite=False):
    embeddings = create_embedding(milvus_openai_embedding_enabled, embedding_name)
    
    if store_in == 'hybrid':
        try:
            with setup_hybrid_db(milvus_host=MILVUS_HOST,
                                 milvus_port=MILVUS_PORT,
                                 milvus_database=MILVUS_DATABASE,
                                 es_url=ES_URL,
                                 # must be lowercase in elasticsearch
                                 collection_or_index_name=table_name.lower(),
                                 embedding=embeddings,
                                 overwrite=overwrite) as hybrid_db:
                
                status = hybrid_db.get_setup_status()
                
                if status == 'skipped':
                    raise BadRequestProblem(
                        detail=f'The table name {table_name} has been used before.'
                               f' If you want to create a new table with the same name, request with overwrite=True.'
                    )
                
                if converter_engine == 'PYMUPDF':
                    one_idp_doc = convert_e_pdf(pdf_bin)
                else:
                    one_idp_doc = convert_scanned_pdf(pdf_bin)

                file_pages = pydash.get(one_idp_doc, ['pages'])
                if not file_pages:
                    raise BadRequestProblem(detail='No content found in input file')
                hybrid_doc = []
                file_by_page = pydash.key_by(file_pages, 'page')
                for page_id, page in file_by_page.items():
                    # Arrange raw text by page
                    if not len(page['paragraphs']):
                        continue
                    raw_doc = create_document_string(page, prefix='', connector=' ')
                    hybrid_doc.append(Document(page_content=raw_doc, metadata={'page': page_id}))

                if len(hybrid_doc) == 0:
                    return 'No content found in input file'

                hybrid_db.insert_documents(hybrid_doc)

                return 'Document finished index building'
        
        except Exception as e:
            return f"ERRORS when building index in hybrid database: {e}"
    
    elif store_in == 'milvus':
        try:
            with setup_milvus_db(milvus_host=MILVUS_HOST,
                                 milvus_port=MILVUS_PORT,
                                 database=MILVUS_DATABASE,
                                 table_name=table_name.lower(),
                                 embedding=embeddings,
                                 overwrite=overwrite) as milvus_db:

                status = milvus_db.get_setup_status()
                
                if status == 'skipped':
                    raise BadRequestProblem(
                        detail=f'The table name {table_name} has been used before.'
                               f' If you want to create a new table with the same name, request with overwrite=True.'
                    )
                
                if converter_engine == 'PYMUPDF':
                    one_idp_doc = convert_e_pdf(pdf_bin)
                else:
                    one_idp_doc = convert_scanned_pdf(pdf_bin)

                file_pages = pydash.get(one_idp_doc, ['pages'])
                if not file_pages:
                    raise BadRequestProblem(detail='No content found in input file')
                milvus_doc = []
                file_by_page = pydash.key_by(file_pages, 'page')
                for page_id, page in file_by_page.items():
                    # Arrange raw text by page
                    if not len(page['paragraphs']):
                        continue
                    raw_doc = create_document_string(page, prefix='', connector=' ')
                    milvus_doc.append(Document(page_content=raw_doc, metadata={'page': page_id}))
                
                if len(milvus_doc) == 0:
                    return 'No content found in input file'
                
                milvus_db.insert_documents(milvus_doc)
                results = milvus_db.search(
                    query="how many shares has been issued during the financial year", top_k=3
                )
                print(results[0])
                return 'Document finished index building in Milvus database'
        
        except Exception as e:
            return f"ERRORS when building index in Milvus database: {e}"
    
    else:
        return "Invalid store_in parameter. Use 'hybrid' or 'milvus'."


if __name__ == "__main__":
    # try building index with financial statement files
    import pathlib
    data_path = pathlib.Path(__file__).parents[0] / 'data'
    pdf_file_path = data_path / 'ATSAR2023+bursa.pdf'
    pdf_bin = pdf_file_path.read_bytes()

    # for local embedding model BgeEmbedding
    table_name = "jsontable_bgeembedding"
    embedding_name = 'BgeEmbeddings'
    converter_engine = 'PYMUPDF'

    build_status = build_document_index(pdf_bin, table_name, converter_engine, store_in='hybrid', milvus_openai_embedding_enabled=True,
                                        embedding_name=embedding_name, overwrite=True)
    print(build_status)


