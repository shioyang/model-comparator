import { ModelComparatorPage } from './app.po';

describe('model-comparator App', () => {
  let page: ModelComparatorPage;

  beforeEach(() => {
    page = new ModelComparatorPage();
  });

  it('should display welcome message', () => {
    page.navigateTo();
    expect(page.getParagraphText()).toEqual('Welcome to app!!');
  });
});
